import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from Nematics3D.logging_decorator import (
    logging_and_warning_decorator,
    set_global_logging_defaults,
)


# Uncomment to set process-wide defaults before running the checks.
# set_global_logging_defaults(
#     log_mode="screen",
#     log_level=10,
#     show_timestamp=False,
# )


@logging_and_warning_decorator(start_finish_level=10)
def safe_divide(label, numerator, denominator, logger=None):
    logger.info(
        f"safe_divide started with label={label}, numerator={numerator}, denominator={denominator}"
    )
    try:
        value = numerator / denominator
    except ZeroDivisionError:
        logger.exception(f"Division by zero inside safe_divide for label={label}")
        denominator = 1
        logger.recovery(f"Recovered by resetting denominator to {denominator}")
        value = numerator / denominator

    return value


@logging_and_warning_decorator(start_finish_level=5)
def decorated_leaf(label, x, y, logger=None):
    logger.info(f"Entered decorated_leaf with label={label}, x={x}, y={y}")
    if y == 0:
        logger.warning(
            f"decorated_leaf detected y=0 for label={label}; recovery path may be triggered"
        )
    result = safe_divide(f"{label}:leaf", x + y, y)
    logger.progress(f"decorated_leaf completed for label={label}")
    return result


def plain_leaf(label, x, y):
    return decorated_leaf(f"{label}:plain-leaf", x, y)


@logging_and_warning_decorator(start_finish_level=10)
def decorated_bridge(label, x, y, logger=None):
    logger.info(f"Entered decorated_bridge with label={label}")
    first = plain_leaf(f"{label}:first-pass", x, y)
    second = decorated_leaf(f"{label}:second-pass", y, x)
    return first + second


def plain_bridge(label, x, y):
    return decorated_bridge(f"{label}:plain-bridge", x, y)


class Sensor:
    def __init__(self, name, bias):
        self.name = name
        self.bias = bias

    @logging_and_warning_decorator(start_finish_level=10)
    def measure(self, raw_x, raw_y, logger=None):
        logger.info(
            f"Sensor.measure received raw_x={raw_x}, raw_y={raw_y}, bias={self.bias}"
        )
        adjusted_x = raw_x + self.bias
        adjusted_y = raw_y - self.bias
        return sensor_plain_middle(self, adjusted_x, adjusted_y)

    def helper_without_decorator(self, x, y):
        return decorated_bridge(f"{self.name}:helper-without-decorator", x, y)


def sensor_plain_middle(sensor, x, y):
    return sensor.helper_without_decorator(x, y)


class Analyzer:
    def __init__(self, name, sensor):
        self.name = name
        self.sensor = sensor

    @logging_and_warning_decorator(start_finish_level=10)
    def analyze(self, x, y, logger=None):
        logger.progress(f"Analyzer.analyze started for name={self.name}")
        if x < y:
            logger.warning(
                f"Analyzer.analyze received x < y for name={self.name}; check input ordering"
            )
        primary = self.sensor.measure(x, y)
        secondary = plain_bridge(f"{self.name}:secondary", y, x)
        logger.info(
            f"Analyzer.analyze collected primary={primary} and secondary={secondary}"
        )
        return self._plain_finalize(primary, secondary)

    def _plain_finalize(self, primary, secondary):
        return build_report(self.name, primary, secondary)


@logging_and_warning_decorator(start_finish_level=10)
def build_report(owner_name, primary, secondary, logger=None):
    logger.info(
        f"Building report for owner_name={owner_name}, primary={primary}, secondary={secondary}"
    )
    total = primary + secondary
    average = total / 2
    return {
        "owner": owner_name,
        "primary": primary,
        "secondary": secondary,
        "total": total,
        "average": average,
    }


class Pipeline:
    def __init__(self, name, analyzer):
        self.name = name
        self.analyzer = analyzer

    @logging_and_warning_decorator(start_finish_level=10)
    def run(self, x, y, logger=None):
        logger.progress(f"Pipeline.run started for pipeline={self.name}")
        report_one = self.analyzer.analyze(x, y)
        report_two = plain_pipeline_step(self.analyzer, x + 1, y - 1)
        logger.info(f"Pipeline.run built two reports for pipeline={self.name}")
        return report_one, report_two


def plain_pipeline_step(analyzer, x, y):
    return analyzer.analyze(x, y)


if __name__ == "__main__":
    sensor = Sensor(name="sensor-alpha", bias=2)
    analyzer = Analyzer(name="analyzer-main", sensor=sensor)
    pipeline = Pipeline(name="pipeline-demo", analyzer=analyzer)

    reports = pipeline.run(
        3,
        0,
        log_mode="screen",
        log_level=30,
        show_timestamp=False,
    )
    print("\nReturned reports:")
    print(reports)
