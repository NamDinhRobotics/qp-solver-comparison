// Native C++ timing harness for the dense QP solvers that have no ready
// Python binding: qpOASES, qpmad and eiquadprog.
//
// It reads the plain-text instances written by export_problems.py, so the
// problems are bit-identical to the ones solved through Python, and reports
// the same statistics (median / p99 of the solve call only). Because nothing
// here crosses a language boundary, these numbers carry no binding overhead
// and are directly comparable to what a C++ control stack would see.
//
// Build: see build_cpp_bench.sh

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <qpmad/solver.h>
#include <qpOASES.hpp>
#include "eiquadprog/eiquadprog-fast.hpp"

using Clock = std::chrono::steady_clock;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct Problem {
    std::string name;
    int n = 0, n_eq = 0, n_in = 0;
    MatrixXd P, A, G;
    VectorXd q, b, h;
};

static Problem load(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    Problem p;
    p.name = path.substr(path.find_last_of('/') + 1);
    p.name = p.name.substr(0, p.name.find_last_of('.'));
    f >> p.n >> p.n_eq >> p.n_in;
    auto rd = [&f](int rows, int cols) {
        MatrixXd M(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) f >> M(i, j);
        return M;
    };
    p.P = rd(p.n, p.n);
    p.q = rd(1, p.n).transpose();
    if (p.n_eq) { p.A = rd(p.n_eq, p.n); p.b = rd(1, p.n_eq).transpose(); }
    p.G = rd(p.n_in, p.n);
    p.h = rd(1, p.n_in).transpose();
    return p;
}

struct Stat { double median = -1, p99 = -1, min_ = -1; int ok = 0; double viol = -1; double obj = 0; };

static Stat summarise(std::vector<double> t, int ok, double viol, double obj) {
    Stat s; s.ok = ok; s.viol = viol; s.obj = obj;
    if (t.empty()) return s;
    std::sort(t.begin(), t.end());
    s.median = t[t.size() / 2];
    s.p99 = t[std::min(t.size() - 1, (size_t)std::llround(0.99 * (t.size() - 1)))];
    s.min_ = t.front();
    return s;
}

// violation of  G x <= h  and  A x = b, relative to the data scale
static double violation(const Problem& p, const VectorXd& x) {
    double v = (p.G * x - p.h).maxCoeff();
    v = std::max(0.0, v) / std::max(1.0, p.h.cwiseAbs().maxCoeff());
    if (p.n_eq) {
        double e = (p.A * x - p.b).cwiseAbs().maxCoeff() /
                   std::max(1.0, p.b.cwiseAbs().maxCoeff());
        v = std::max(v, e);
    }
    return v;
}

static double objective(const Problem& p, const VectorXd& x) {
    return 0.5 * x.dot(p.P * x) + p.q.dot(x);
}

// ---------------------------------------------------------------- qpmad
static Stat run_qpmad(const Problem& p, int reps) {
    const double INF = std::numeric_limits<double>::infinity();
    const int m = p.n_eq + p.n_in;
    MatrixXd A(m, p.n);
    VectorXd lbA(m), ubA(m);
    if (p.n_eq) { A.topRows(p.n_eq) = p.A; lbA.head(p.n_eq) = p.b; ubA.head(p.n_eq) = p.b; }
    A.bottomRows(p.n_in) = p.G;
    lbA.tail(p.n_in).setConstant(-INF);
    ubA.tail(p.n_in) = p.h;

    std::vector<double> ts;
    VectorXd x;
    int ok = 0;
    for (int r = 0; r < reps + 3; ++r) {
        MatrixXd H = p.P;          // qpmad factorises H in place
        VectorXd g = p.q;
        qpmad::Solver solver;
        auto t0 = Clock::now();
        auto st = solver.solve(x, H, g, A, lbA, ubA);
        double dt = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (r >= 3) { ts.push_back(dt); if (st == qpmad::Solver::OK) ++ok; }
    }
    return summarise(ts, ok, violation(p, x), objective(p, x));
}

// ------------------------------------------------------------ eiquadprog
static Stat run_eiquadprog(const Problem& p, int reps) {
    // solves min .5 x'Qx + c'x  s.t.  Aeq x + beq = 0,  Cin x + din >= 0
    MatrixXd Aeq = p.n_eq ? p.A : MatrixXd(0, p.n);
    VectorXd beq = p.n_eq ? VectorXd(-p.b) : VectorXd(0);
    MatrixXd Cin = -p.G;
    VectorXd din = p.h;

    std::vector<double> ts;
    VectorXd x(p.n);
    int ok = 0;
    for (int r = 0; r < reps + 3; ++r) {
        eiquadprog::solvers::EiquadprogFast qp;
        qp.reset(p.n, Aeq.rows(), Cin.rows());
        MatrixXd Q = p.P;
        VectorXd c = p.q;
        auto t0 = Clock::now();
        auto st = qp.solve_quadprog(Q, c, Aeq, beq, Cin, din, x);
        double dt = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (r >= 3) {
            ts.push_back(dt);
            if (st == eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) ++ok;
        }
    }
    return summarise(ts, ok, violation(p, x), objective(p, x));
}

// --------------------------------------------------------------- qpOASES
// cold: a fresh QProblem per solve (comparable to every other cold number)
static Stat run_qpoases(const Problem& p, int reps, bool hotstart, Stat* hot,
                        bool mpc_preset) {
    using namespace qpOASES;
    const int m = p.n_eq + p.n_in;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Hr = p.P, Ar(m, p.n);
    if (p.n_eq) Ar.topRows(p.n_eq) = p.A;
    Ar.bottomRows(p.n_in) = p.G;
    VectorXd lbA(m), ubA(m);
    if (p.n_eq) { lbA.head(p.n_eq) = p.b; ubA.head(p.n_eq) = p.b; }
    lbA.tail(p.n_in).setConstant(-INFTY);
    ubA.tail(p.n_in) = p.h;

    Options opt;
    if (mpc_preset) opt.setToMPC();   // the preset qpOASES ships for MPC
    opt.printLevel = PL_NONE;
    std::vector<double> ts;
    VectorXd x(p.n);
    int ok = 0;
    for (int r = 0; r < reps + 3; ++r) {
        QProblem qp(p.n, m);
        qp.setOptions(opt);
        int_t nWSR = 10000;
        auto t0 = Clock::now();
        returnValue rv = qp.init(Hr.data(), p.q.data(), Ar.data(),
                                 nullptr, nullptr, lbA.data(), ubA.data(), nWSR);
        double dt = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (rv == SUCCESSFUL_RETURN) qp.getPrimalSolution(x.data());
        if (r >= 3) { ts.push_back(dt); if (rv == SUCCESSFUL_RETURN) ++ok; }
    }
    Stat cold = summarise(ts, ok, violation(p, x), objective(p, x));

    if (hotstart && hot) {
        // the parametric warm start qpOASES is known for: keep the object,
        // perturb the linear term, re-solve with hotstart()
        QProblem qp(p.n, m);
        qp.setOptions(opt);
        int_t nWSR = 10000;
        qp.init(Hr.data(), p.q.data(), Ar.data(), nullptr, nullptr,
                lbA.data(), ubA.data(), nWSR);
        std::vector<double> th;
        int okh = 0;
        VectorXd g = p.q;
        for (int r = 0; r < reps + 3; ++r) {
            for (int i = 0; i < p.n; ++i)
                g[i] = p.q[i] * (1.0 + 1e-3 * std::sin(0.1 * r + i));
            nWSR = 10000;
            auto t0 = Clock::now();
            returnValue rv = qp.hotstart(g.data(), nullptr, nullptr,
                                         lbA.data(), ubA.data(), nWSR);
            double dt = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            if (r >= 3) { th.push_back(dt); if (rv == SUCCESSFUL_RETURN) ++okh; }
        }
        qp.getPrimalSolution(x.data());
        *hot = summarise(th, okh, violation(p, x), objective(p, x));
    }
    return cold;
}

static void emit(const std::string& inst, const std::string& solver, const Stat& s,
                 bool& first) {
    if (!first) std::cout << ",\n";
    first = false;
    std::cout << "  {\"instance\": \"" << inst << "\", \"solver\": \"" << solver
              << "\", \"median_ms\": " << s.median
              << ", \"p99_ms\": " << s.p99
              << ", \"min_ms\": " << s.min_
              << ", \"n_ok\": " << s.ok
              << ", \"rel_violation\": " << s.viol
              << ", \"objective\": " << s.obj << "}";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: cpp_bench <reps> <file1.txt> [file2.txt ...]\n";
        return 1;
    }
    const int reps = std::stoi(argv[1]);
    std::cout.precision(10);
    std::cout << "[\n";
    bool first = true;
    for (int i = 2; i < argc; ++i) {
        Problem p = load(argv[i]);
        std::cerr << "-- " << p.name << " n=" << p.n
                  << " n_eq=" << p.n_eq << " n_in=" << p.n_in << "\n";
        emit(p.name, "qpmad", run_qpmad(p, reps), first);
        emit(p.name, "eiquadprog", run_eiquadprog(p, reps), first);
        Stat hot_def, hot_mpc;
        emit(p.name, "qpOASES", run_qpoases(p, reps, true, &hot_def, false), first);
        emit(p.name, "qpOASES_hotstart", hot_def, first);
        emit(p.name, "qpOASES_mpc", run_qpoases(p, reps, true, &hot_mpc, true), first);
        emit(p.name, "qpOASES_mpc_hotstart", hot_mpc, first);
    }
    std::cout << "\n]\n";
    return 0;
}
