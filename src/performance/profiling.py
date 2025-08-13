"""Comprehensive performance profiling and monitoring system."""

import cProfile
import io
import json
import logging
import os
import pstats
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import psutil

try:
    import py_spy

    PY_SPY_AVAILABLE = True
except ImportError:
    py_spy = None
    PY_SPY_AVAILABLE = False

try:
    from line_profiler import LineProfiler

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LineProfiler = None
    LINE_PROFILER_AVAILABLE = False

try:
    import memory_profiler

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    memory_profiler = None
    MEMORY_PROFILER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of a profiling session."""

    function_name: str
    total_time: float
    calls_count: int
    avg_time_per_call: float
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    detailed_stats: Dict[str, Any] = field(default_factory=dict)
    flamegraph_data: Optional[str] = None


@dataclass
class PerformanceHotspot:
    """Identified performance hotspot."""

    location: str
    function_name: str
    line_number: int
    time_percentage: float
    cumulative_time: float
    calls_count: int
    severity: str  # low, medium, high, critical


class FlameGraphGenerator:
    """Generate flame graphs for performance visualization."""

    def __init__(self):
        self.stack_samples = []
        self.sampling_active = False
        self.sampling_thread = None
        self.sampling_interval = 0.01  # 10ms

        logger.info("Flame graph generator initialized")

    def start_sampling(self):
        """Start stack sampling for flame graph generation."""
        if self.sampling_active:
            return

        self.sampling_active = True
        self.stack_samples = []
        self.sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self.sampling_thread.start()

        logger.info("Started flame graph sampling")

    def stop_sampling(self):
        """Stop stack sampling."""
        if not self.sampling_active:
            return

        self.sampling_active = False
        if self.sampling_thread:
            self.sampling_thread.join(timeout=1.0)

        logger.info(f"Stopped flame graph sampling ({len(self.stack_samples)} samples)")

    def _sampling_loop(self):
        """Main sampling loop."""
        import sys
        import traceback

        while self.sampling_active:
            try:
                # Capture current stack trace
                frame = sys._current_frames()
                for thread_id, frame_obj in frame.items():
                    stack = []
                    current_frame = frame_obj

                    while current_frame:
                        filename = current_frame.f_code.co_filename
                        function_name = current_frame.f_code.co_name
                        line_number = current_frame.f_lineno

                        stack.append(
                            f"{os.path.basename(filename)}:{function_name}:{line_number}"
                        )
                        current_frame = current_frame.f_back

                    if stack:
                        # Reverse to get root->leaf order
                        stack.reverse()
                        stack_trace = ";".join(stack)
                        self.stack_samples.append(stack_trace)

                time.sleep(self.sampling_interval)

            except Exception as e:
                logger.error(f"Sampling error: {e}")
                time.sleep(self.sampling_interval)

    def generate_flamegraph_data(self) -> str:
        """Generate flame graph data in folded stack format."""
        if not self.stack_samples:
            return ""

        # Count stack occurrences
        stack_counts = defaultdict(int)
        for stack in self.stack_samples:
            stack_counts[stack] += 1

        # Generate folded format
        folded_lines = []
        for stack, count in stack_counts.items():
            folded_lines.append(f"{stack} {count}")

        return "\n".join(folded_lines)

    def save_flamegraph(self, output_path: str, title: str = "Performance Profile"):
        """Save flame graph to file."""
        folded_data = self.generate_flamegraph_data()

        if not folded_data:
            logger.warning("No sampling data available for flame graph")
            return False

        try:
            # Save folded data
            folded_path = output_path.replace(".html", ".folded")
            with open(folded_path, "w") as f:
                f.write(folded_data)

            # Generate HTML flame graph (simplified version)
            html_content = self._generate_html_flamegraph(folded_data, title)

            with open(output_path, "w") as f:
                f.write(html_content)

            logger.info(f"Flame graph saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save flame graph: {e}")
            return False

    def _generate_html_flamegraph(self, folded_data: str, title: str) -> str:
        """Generate HTML flame graph visualization."""
        # Simplified HTML flame graph template
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .flamegraph {{ background: #f0f0f0; padding: 20px; border-radius: 8px; }}
        .stack-frame {{ 
            display: block; 
            margin: 2px 0; 
            padding: 4px 8px; 
            background: #e0e0e0; 
            border-radius: 4px; 
        }}
        .hot {{ background: #ff6b6b; }}
        .warm {{ background: #ffd93d; }}
        .cool {{ background: #6bcf7f; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="flamegraph">
        <h2>Stack Traces (Top Functions)</h2>
        <div id="stacks">
        {self._format_stacks_html(folded_data)}
        </div>
    </div>
    <div>
        <h2>Raw Data</h2>
        <pre>{folded_data[:2000]}{"..." if len(folded_data) > 2000 else ""}</pre>
    </div>
</body>
</html>
"""
        return html_template

    def _format_stacks_html(self, folded_data: str) -> str:
        """Format stack data as HTML."""
        lines = folded_data.split("\n")[:50]  # Top 50 stacks
        html_parts = []

        for line in lines:
            if not line.strip():
                continue

            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue

            stack, count = parts[0], int(parts[1])

            # Color based on count
            css_class = "cool"
            if count > 10:
                css_class = "warm"
            if count > 50:
                css_class = "hot"

            html_parts.append(
                f'<div class="stack-frame {css_class}">'
                f'<strong>{count}</strong>: {stack.split(";")[-1]}'
                f"</div>"
            )

        return "\n".join(html_parts)


class DetailedProfiler:
    """Detailed performance profiler with multiple profiling backends."""

    def __init__(self):
        self.active_profilers = []
        self.profile_results = {}
        self.hotspot_detector = HotspotDetector()
        self.flamegraph_generator = FlameGraphGenerator()

        # Check available profilers
        self.available_profilers = {
            "cProfile": True,
            "line_profiler": LINE_PROFILER_AVAILABLE,
            "memory_profiler": MEMORY_PROFILER_AVAILABLE,
            "py_spy": PY_SPY_AVAILABLE,
        }

        logger.info(
            f"Detailed profiler initialized with backends: {list(self.available_profilers.keys())}"
        )

    def profile_function(
        self,
        func: Callable,
        *args,
        profiler_types: List[str] = None,
        enable_flamegraph: bool = True,
        **kwargs,
    ) -> ProfileResult:
        """Profile a single function execution."""
        if profiler_types is None:
            profiler_types = ["cProfile"]

        function_name = f"{func.__module__}.{func.__name__}"

        # Start profiling
        profilers = {}

        # cProfile
        if "cProfile" in profiler_types:
            profilers["cProfile"] = cProfile.Profile()
            profilers["cProfile"].enable()

        # Line profiler
        if "line_profiler" in profiler_types and LINE_PROFILER_AVAILABLE:
            profilers["line_profiler"] = LineProfiler()
            profilers["line_profiler"].add_function(func)
            profilers["line_profiler"].enable()

        # Memory profiler (start monitoring)
        memory_usage_before = psutil.Process().memory_info().rss / (1024 * 1024)

        # Flame graph sampling
        if enable_flamegraph:
            self.flamegraph_generator.start_sampling()

        # Execute function
        start_time = time.time()
        start_cpu_time = psutil.Process().cpu_times()

        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            logger.error(f"Function execution failed during profiling: {e}")

        end_time = time.time()
        end_cpu_time = psutil.Process().cpu_times()

        # Stop profiling
        total_time = end_time - start_time
        cpu_time = (end_cpu_time.user - start_cpu_time.user) + (
            end_cpu_time.system - start_cpu_time.system
        )
        cpu_usage = (cpu_time / total_time * 100) if total_time > 0 else 0

        if enable_flamegraph:
            self.flamegraph_generator.stop_sampling()
            flamegraph_data = self.flamegraph_generator.generate_flamegraph_data()
        else:
            flamegraph_data = None

        memory_usage_after = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_delta = memory_usage_after - memory_usage_before

        # Process profiling results
        detailed_stats = {}
        calls_count = 1

        # Process cProfile results
        if "cProfile" in profilers:
            profilers["cProfile"].disable()

            s = io.StringIO()
            ps = pstats.Stats(profilers["cProfile"], stream=s)
            ps.sort_stats("cumulative")
            ps.print_stats(20)  # Top 20 functions

            detailed_stats["cProfile"] = s.getvalue()

            # Get call count for this function
            stats = ps.get_stats()
            for (filename, line, func_name), (cc, nc, tt, ct) in stats.items():
                if func_name == func.__name__:
                    calls_count = cc
                    break

        # Process line profiler results
        if "line_profiler" in profilers:
            profilers["line_profiler"].disable()

            s = io.StringIO()
            profilers["line_profiler"].print_stats(stream=s)
            detailed_stats["line_profiler"] = s.getvalue()

        # Create profile result
        avg_time = total_time / calls_count if calls_count > 0 else total_time

        profile_result = ProfileResult(
            function_name=function_name,
            total_time=total_time,
            calls_count=calls_count,
            avg_time_per_call=avg_time,
            memory_usage_mb=memory_delta,
            cpu_usage_percent=cpu_usage,
            detailed_stats=detailed_stats,
            flamegraph_data=flamegraph_data,
        )

        # Store result
        self.profile_results[function_name] = profile_result

        # Detect hotspots
        if "cProfile" in profilers:
            hotspots = self.hotspot_detector.analyze_profile(profilers["cProfile"])
            profile_result.detailed_stats["hotspots"] = [
                {
                    "location": h.location,
                    "function_name": h.function_name,
                    "time_percentage": h.time_percentage,
                    "severity": h.severity,
                }
                for h in hotspots
            ]

        logger.info(
            f"Profiled {function_name}: {total_time:.4f}s, {memory_delta:.2f}MB, {cpu_usage:.1f}% CPU"
        )

        return profile_result

    def profile_context(
        self,
        context_name: str,
        profiler_types: List[str] = None,
        enable_flamegraph: bool = True,
    ):
        """Context manager for profiling a code block."""

        class ProfileContext:
            def __init__(self, profiler, name, types, flamegraph):
                self.profiler = profiler
                self.context_name = name
                self.profiler_types = types or ["cProfile"]
                self.enable_flamegraph = flamegraph
                self.start_time = None
                self.active_profilers = {}

            def __enter__(self):
                self.start_time = time.time()

                # Start profilers
                if "cProfile" in self.profiler_types:
                    self.active_profilers["cProfile"] = cProfile.Profile()
                    self.active_profilers["cProfile"].enable()

                if self.enable_flamegraph:
                    self.profiler.flamegraph_generator.start_sampling()

                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                total_time = end_time - self.start_time

                # Stop profilers
                if "cProfile" in self.active_profilers:
                    self.active_profilers["cProfile"].disable()

                if self.enable_flamegraph:
                    self.profiler.flamegraph_generator.stop_sampling()
                    flamegraph_data = (
                        self.profiler.flamegraph_generator.generate_flamegraph_data()
                    )
                else:
                    flamegraph_data = None

                # Process results
                detailed_stats = {}
                if "cProfile" in self.active_profilers:
                    s = io.StringIO()
                    ps = pstats.Stats(self.active_profilers["cProfile"], stream=s)
                    ps.sort_stats("cumulative")
                    ps.print_stats(20)
                    detailed_stats["cProfile"] = s.getvalue()

                # Create profile result
                profile_result = ProfileResult(
                    function_name=self.context_name,
                    total_time=total_time,
                    calls_count=1,
                    avg_time_per_call=total_time,
                    detailed_stats=detailed_stats,
                    flamegraph_data=flamegraph_data,
                )

                # Store result
                self.profiler.profile_results[self.context_name] = profile_result

                logger.info(
                    f"Context profiled '{self.context_name}': {total_time:.4f}s"
                )

        return ProfileContext(self, context_name, profiler_types, enable_flamegraph)

    def save_profile_report(self, output_dir: str, include_flamegraphs: bool = True):
        """Save comprehensive profiling report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate main report
        report = {
            "timestamp": time.time(),
            "profiling_summary": {
                "total_functions_profiled": len(self.profile_results),
                "available_profilers": self.available_profilers,
            },
            "profile_results": {},
        }

        # Process each profile result
        for func_name, result in self.profile_results.items():
            result_data = {
                "function_name": result.function_name,
                "total_time": result.total_time,
                "calls_count": result.calls_count,
                "avg_time_per_call": result.avg_time_per_call,
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_usage_percent": result.cpu_usage_percent,
                "detailed_stats_keys": list(result.detailed_stats.keys()),
            }

            # Save detailed stats to separate files
            for stat_type, stat_data in result.detailed_stats.items():
                stat_filename = f"{func_name.replace('.', '_')}_{stat_type}.txt"
                stat_path = output_path / stat_filename

                with open(stat_path, "w") as f:
                    if isinstance(stat_data, str):
                        f.write(stat_data)
                    else:
                        f.write(json.dumps(stat_data, indent=2))

                result_data[f"{stat_type}_file"] = stat_filename

            # Save flame graph
            if include_flamegraphs and result.flamegraph_data:
                flamegraph_filename = f"{func_name.replace('.', '_')}_flamegraph.html"
                flamegraph_path = output_path / flamegraph_filename

                # Create temporary flame graph generator for this data
                temp_generator = FlameGraphGenerator()
                temp_generator.stack_samples = result.flamegraph_data.split("\n")
                temp_generator.save_flamegraph(
                    str(flamegraph_path), f"Profile: {func_name}"
                )

                result_data["flamegraph_file"] = flamegraph_filename

            report["profile_results"][func_name] = result_data

        # Save main report
        report_path = output_path / "profiling_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Profiling report saved to {output_dir}")
        return str(report_path)

    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling results."""
        if not self.profile_results:
            return {"message": "No profiling data available"}

        total_time = sum(r.total_time for r in self.profile_results.values())
        total_calls = sum(r.calls_count for r in self.profile_results.values())

        # Find slowest functions
        slowest_functions = sorted(
            self.profile_results.items(), key=lambda x: x[1].total_time, reverse=True
        )[:5]

        # Find memory-heavy functions
        memory_heavy = sorted(
            self.profile_results.items(),
            key=lambda x: x[1].memory_usage_mb,
            reverse=True,
        )[:5]

        return {
            "total_functions_profiled": len(self.profile_results),
            "total_execution_time": total_time,
            "total_function_calls": total_calls,
            "average_time_per_function": total_time / len(self.profile_results),
            "slowest_functions": [
                {"name": name, "time": result.total_time, "calls": result.calls_count}
                for name, result in slowest_functions
            ],
            "memory_heavy_functions": [
                {
                    "name": name,
                    "memory_mb": result.memory_usage_mb,
                    "time": result.total_time,
                }
                for name, result in memory_heavy
            ],
        }


class HotspotDetector:
    """Detect performance hotspots in profiling data."""

    def __init__(self, time_threshold_percent: float = 5.0, calls_threshold: int = 100):
        self.time_threshold_percent = time_threshold_percent
        self.calls_threshold = calls_threshold

    def analyze_profile(self, profiler: cProfile.Profile) -> List[PerformanceHotspot]:
        """Analyze cProfile data to find hotspots."""
        stats = pstats.Stats(profiler)
        hotspots = []

        total_time = stats.total_tt

        for (filename, line_number, function_name), (
            cc,
            nc,
            tt,
            ct,
        ) in stats.get_stats().items():
            time_percentage = (ct / total_time * 100) if total_time > 0 else 0

            # Determine severity
            severity = "low"
            if time_percentage > 20:
                severity = "critical"
            elif time_percentage > 10:
                severity = "high"
            elif time_percentage > 5:
                severity = "medium"

            # Only include significant hotspots
            if (
                time_percentage >= self.time_threshold_percent
                or cc >= self.calls_threshold
            ):
                hotspot = PerformanceHotspot(
                    location=f"{os.path.basename(filename)}:{line_number}",
                    function_name=function_name,
                    line_number=line_number,
                    time_percentage=time_percentage,
                    cumulative_time=ct,
                    calls_count=cc,
                    severity=severity,
                )
                hotspots.append(hotspot)

        # Sort by time percentage
        hotspots.sort(key=lambda x: x.time_percentage, reverse=True)

        return hotspots[:20]  # Top 20 hotspots


# Global profiler instance
_global_profiler = None


def get_profiler() -> DetailedProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = DetailedProfiler()
    return _global_profiler


def profile_performance(
    profiler_types: List[str] = None,
    enable_flamegraph: bool = True,
    save_results: bool = False,
    output_dir: str = "./profiling_results",
):
    """Decorator for performance profiling."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()

            result = profiler.profile_function(
                func,
                *args,
                profiler_types=profiler_types,
                enable_flamegraph=enable_flamegraph,
                **kwargs,
            )

            if save_results:
                profiler.save_profile_report(
                    output_dir, include_flamegraphs=enable_flamegraph
                )

            # Return original function result, not profile result
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Function execution failed: {e}")
                raise

        return wrapper

    return decorator


def profile_context(
    context_name: str, profiler_types: List[str] = None, enable_flamegraph: bool = True
):
    """Context manager for profiling code blocks."""
    profiler = get_profiler()
    return profiler.profile_context(context_name, profiler_types, enable_flamegraph)
