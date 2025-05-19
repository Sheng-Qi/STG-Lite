from typing import Type, NamedTuple, Self, List, Protocol
import math
import numpy as np

class Interval(NamedTuple):
    lower: float
    upper : float

    def divide(self, n):
        """
        Divide this interval to n equal intervals
        """
        steps = np.linspace(self.lower, self.upper, n)
        return [Interval(steps[i], steps[i+1]) for i in range(len(steps)-1)] 

    def get_duration(self):
        return self.lower + self.upper

    def is_within(self, time : float):
        return self.lower <= time < self.upper

    def contains(self, other : Self):
        """
        Checks if this interval fully contains the other interval

        e.g [0.0, 1.0] contains [0.3,0.7] -> yes
        """
        return self.lower  <= other.lower and self.upper > other.upper

    def get_midpoint(self):
        """
        Get midpoint of the interval
        """
        return 0.5 * self.get_duration()

class Segment(Protocol):
    interval : Interval

    def push(self, item) -> None:
        ...

class Gaussian(Segment):
    def __init__(self, interval : Interval):
        self.interval = interval

    def push(self, item) -> None:
        ...

class SegmentTree:
    def __init__(self, max_level : int, duration : float, segment_class : Type[Segment]):
        self.max_level = max_level
        self.duration = duration
        self.max_interval = Interval(0.0, self.duration)

        self.segments : List[Segment] = []
        for i in range(max_level+1):
            num_segments = 2**i
            intervals = self.max_interval.divide(num_segments+1)
            for interval in intervals:
                self.segments.append(segment_class(interval))
        print(f"Created segment tree with max level {self.max_level} and {len(self.segments)} segments")

    def push(self, item_interval : Interval, item, verbose=False):
        def dfs(cur_segment_id : int, cur_interval : Interval):
            if not cur_interval.contains(item_interval):
                if cur_segment_id == 0:
                    return 0
                return -1
            
            if cur_segment_id >= len(self.segments):
                return -1
            
            mid = cur_interval.get_midpoint()

            left_segment_interval = Interval(cur_interval.lower, mid)
            left_segment_id = dfs(1 + cur_segment_id * 2, left_segment_interval)
            if left_segment_id != -1:
                return left_segment_id

            right_segment_interval = Interval(mid, cur_interval.upper)
            right_segment_id = dfs(2 + cur_segment_id * 2, right_segment_interval)
            if right_segment_id != -1:
                return right_segment_id

            return cur_segment_id

        id = dfs(0, self.max_interval)
        if verbose:
            print(f"{item_interval} pushed to {id}->{self.segments[id].interval}")
        self.segments[id].push(item)

    def get_at_time(self, time : float) -> List[Segment]:
        if not self.max_interval.is_within(time):
            print(f"Failed to query segments, {time} is not within the max interval {self.max_interval}")
            return []
        segments_out = [self.segments[0]]

        cur_level_duration = self.max_interval.get_duration()
        for level in range(1, self.max_level+1):
            cur_level_duration /= 2.0
            cur_level_segment_id = math.floor(time / cur_level_duration)
            segment_linear_id = 2 ** level + cur_level_segment_id - 1
            segments_out.append(self.segments[segment_linear_id])
        return segments_out

    
        
if __name__ == "__main__":
    st = SegmentTree(3, 1.0, Gaussian)
    st.push(Interval(-0.3, 100.0), None, verbose=True)
    st.push(Interval(0.3, 0.7), None, verbose=True)

    st.push(Interval(0.1, 0.2), None, verbose=True)
    st.push(Interval(0.3, 0.4), None, verbose=True)

    st.push(Interval(0.0, 0.49), None, verbose=True)

    st.push(Interval(0.6, 0.7), None, verbose=True)
    st.push(Interval(0.8, 0.9), None, verbose=True)

    st.push(Interval(0.0, 0.12), None, verbose=True)
    st.push(Interval(0.13, 0.24), None, verbose=True)

    st.push(Interval(0.76, 0.87), None, verbose=True)
    st.push(Interval(0.88, 0.99), None, verbose=True)

    print([seg.interval for seg in st.get_at_time(0.5)])