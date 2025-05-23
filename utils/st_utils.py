from typing import Type, NamedTuple, Protocol
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

    def get_duration(self) -> float:
        """
        Returns lower + upper
        """
        return self.lower + self.upper

    def is_within(self, time : float) -> bool:
        """
        Check if time is inside the interval
        """
        return self.lower <= time < self.upper

    def __str__(self) -> str:
        return f"[{self.lower:.3f}, {self.upper:.3f}]"

class Segment(Protocol):
    interval : Interval

    def size() -> int:
        """
        Returns the size of segment
        """
        ...

class SegmentTree:
    def __init__(self, max_level : int, duration : float, segment_class : Type[Segment]):
        self.max_level = max_level
        self.duration = duration
        self.max_interval = Interval(0.0, self.duration)

        self.segments : list[Segment] = []
        for i in range(max_level+1):
            num_segments = 2**i
            intervals = self.max_interval.divide(num_segments+1)
            for interval in intervals:
                self.segments.append(segment_class(interval))
        print(f"Created segment tree with max level {self.max_level} and {len(self.segments)} segments")

    def get_all_segments_ref(self) -> list[Segment]:
        return self.segments
    
    def get_active_gaussians_id_at_time(self, time : float) -> list:
        if not self.max_interval.is_within(time):
            print(f"Failed to query segments, {time} is not within the max interval {self.max_interval}")
            return []
        
        active_gaussian_ids = [self.segments[0].gaussian_ids]

        cur_level_duration = self.max_interval.get_duration()
        for level in range(1, self.max_level+1):
            cur_level_duration /= 2.0
            cur_level_segment_id = math.floor(time / cur_level_duration)
            segment_linear_id = int(2 ** level + cur_level_segment_id - 1)
            segment_active_gaussian_ids = self.segments[segment_linear_id].gaussian_ids
            active_gaussian_ids.append(segment_active_gaussian_ids)
        return active_gaussian_ids
    
    
        
if __name__ == "__main__":
    class Gaussian(Segment):
        def __init__(self, interval : Interval):
            self.interval = interval

        def push(self, item) -> None:
            ...
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