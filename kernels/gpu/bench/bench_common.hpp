// Small helpers shared by the benchmarks.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace bench {

inline void hip_check(hipError_t err, const char *what) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
    }
}

// "1,128,512" -> {1, 128, 512}
inline std::vector<int64_t> parse_list(const std::string &s) {
    std::vector<int64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::stoll(item));
        }
    }
    return out;
}

// Value of "--name v" or the default.
inline std::string arg(int argc, char **argv, const char *name, const std::string &def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return def;
}

inline bool flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return true;
        }
    }
    return false;
}

inline double median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Runs `work` (GPU work enqueued on `stream`) again and again for `seconds`, so that the GPU clock is up before
// timing. From idle the HX 370's GPU clock takes seconds to climb from about 600 MHz to about 2.7 GHz.
template <class Work> void warm_up_clock(hipStream_t stream, double seconds, Work &&work) {
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
        for (int i = 0; i < 10; ++i) {
            work();
        }
        hip_check(hipStreamSynchronize(stream), "warm_up_clock");
    }
}

// Records events on a stream and returns the elapsed milliseconds between two marks.
// Record all marks first and call ms() after the loop. A sync inside the loop leaves the GPU idle
// at the next mark, so the host launch time gets counted as kernel time.
class EventTimer {
  public:
    EventTimer(hipStream_t stream, int marks) : stream_(stream), events_(marks) {
        for (auto &e : events_) {
            hip_check(hipEventCreate(&e), "hipEventCreate");
        }
    }
    ~EventTimer() {
        for (auto &e : events_) {
            (void)hipEventDestroy(e);
        }
    }
    void mark(int i) { hip_check(hipEventRecord(events_[i], stream_), "hipEventRecord"); }
    float ms(int from, int to) {
        hip_check(hipEventSynchronize(events_[to]), "hipEventSynchronize");
        float t = 0.0f;
        hip_check(hipEventElapsedTime(&t, events_[from], events_[to]), "hipEventElapsedTime");
        return t;
    }

  private:
    hipStream_t stream_;
    std::vector<hipEvent_t> events_;
};

// Samples the GPU's hwmon sensors (read-only) on a background thread:
//   freq1_input  GPU clock (Hz)       power1_input  package power PPT of the whole SoC (uW)
//   temp1_input  GPU edge temperature (millidegree C)
// Call start() before the GPU work and stop() after it. Values are -1 when the files are missing.
class SensorSampler {
  public:
    struct Stats {
        double sclk_mhz_median = -1, sclk_mhz_min = -1, power_w_median = -1, temp_c_max = -1;
    };

    explicit SensorSampler(int period_ms = 5) : period_ms_(period_ms) {
        namespace fs = std::filesystem;
        std::error_code ec;
        for (const auto &card : fs::directory_iterator("/sys/class/drm", ec)) {
            const fs::path hw = card.path() / "device" / "hwmon";
            if (!fs::exists(hw, ec)) {
                continue;
            }
            for (const auto &h : fs::directory_iterator(hw, ec)) {
                if (fs::exists(h.path() / "freq1_input", ec)) {
                    dir_ = h.path().string();
                    return;
                }
            }
        }
    }
    ~SensorSampler() { stop(); }

    bool available() const { return !dir_.empty(); }

    void start() {
        stop();
        if (!available()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            sclk_.clear(), power_.clear(), temp_.clear();
        }
        running_ = true;
        thread_ = std::thread([this] {
            while (running_) {
                const double f = read(dir_ + "/freq1_input"), p = read(dir_ + "/power1_input"), t = read(dir_ + "/temp1_input");
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    if (f >= 0) sclk_.push_back(f / 1e6);
                    if (p >= 0) power_.push_back(p / 1e6);
                    if (t >= 0) temp_.push_back(t / 1e3);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(period_ms_));
            }
        });
    }

    Stats stop() {
        if (thread_.joinable()) {
            running_ = false;
            thread_.join();
        }
        std::lock_guard<std::mutex> lock(mu_);
        Stats s;
        if (!sclk_.empty()) {
            s.sclk_mhz_median = median(sclk_);
            s.sclk_mhz_min = *std::min_element(sclk_.begin(), sclk_.end());
        }
        if (!power_.empty()) s.power_w_median = median(power_);
        if (!temp_.empty()) s.temp_c_max = *std::max_element(temp_.begin(), temp_.end());
        return s;
    }

  private:
    static double read(const std::string &path) {
        std::ifstream f(path);
        double v = -1;
        return (f >> v) ? v : -1;
    }

    std::string dir_;
    int period_ms_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mu_;
    std::vector<double> sclk_, power_, temp_;
};

} // namespace bench
