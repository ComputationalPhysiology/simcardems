import math
import typing


class TimeStepper:
    def __init__(
        self,
        *,
        t0: float,
        T: float,
        dt: float,
        use_ns: bool = True,
        st_progress: typing.Any = None,
    ) -> None:
        """Initialize time stepper

        Parameters
        ----------
        t0 : float
            Start time in milliseconds
        T : float
            End time in milliseconds
        dt : float
            Time step
        use_ns : bool, optional
            Whether to return the time in nanoseconds, by default True
        st_progress:
            Streamlit progress bar
        """

        self._use_ns = use_ns
        self._st_progress = st_progress

        if use_ns:
            self.t0 = TimeStepper.ms2ns(t0)
            self.T = TimeStepper.ms2ns(T)
            self.dt = TimeStepper.ms2ns(dt)
        else:
            self.t0 = t0
            self.T = T
            self.dt = dt

        self.reset()

    def reset(self):
        self.t = self.t0
        self.step = 0

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, T: float) -> None:
        if self.t0 > T:
            raise ValueError("Start time has to be lower than end time")
        self._T = T

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        dt = min(self.T - self.t0, dt)
        self._dt = dt

    @property
    def total_steps(self) -> int:
        if math.isclose(self.T, self.t0):
            return 0
        return round((self.T - self.t0) / self.dt)

    def __iter__(self):
        if self.T is None:
            raise RuntimeError("Please assign an end time to time stepper")
        while self.t < self.T:
            prev_t = self.t
            self.t = min(self.t + self.dt, self.T)
            self.step += 1
            if self._st_progress is not None:
                self._st_progress.progress(self.step / self.total_steps)
            yield prev_t, self.t

    @staticmethod
    def ns2ms(t: float) -> float:
        """Convert nanoseconds to milliseconds

        Parameters
        ----------
        t : float
            The time in nanoseconds

        Returns
        -------
        float
            Time in milliseconds
        """
        return t * 1e-6

    @staticmethod
    def ms2ns(t: float) -> float:
        """Convert from milliseconds to nanoseconds

        Parameters
        ----------
        t : float
            Time in milliseconds

        Returns
        -------
        float
            Time in nanoseconds
        """
        return int(t * 1e6)
