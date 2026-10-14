// ALIVE-LIO: Degeneracy-Aware Learning of Inertial Velocity for Enhancing
// ESKF-Based LiDAR-Inertial Odometry (IEEE T-ITS 2026)
// Copyright (c) 2026 Seongjun Kim, POSTECH.
// Licensed under the GNU General Public License v2.0 (see LICENSE).

template<typename T>
class FixedDeque {
    std::deque<T> dq;
    size_t max_len;

public:
    FixedDeque(size_t max_len_) : max_len(max_len_) {}

    void push_back(const T& value) {
        dq.push_back(value);
        if (dq.size() > max_len)
            dq.pop_front();
    }

    void resize(const size_t num_seq) {
        max_len = num_seq;
    }

    size_t size() const { return dq.size(); }
    const T& front() const { return dq.front(); }
    const T& back() const { return dq.back(); }
    const std::deque<T>& data() const { return dq; }
};