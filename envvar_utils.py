import os

def log_lb_enabled() -> bool:
    LOG_LOAD_BALANCE = os.getenv('LOG_LOAD_BALANCE', False)
    return LOG_LOAD_BALANCE


def is_in_eval() -> bool:
    glue_evaluating = os.getenv("glue_evaluating", False)
    cloth_evaluating = os.getenv("cloth_evaluating", False)
    squad_evaluating = os.getenv("squad_evaluating", False)
    return glue_evaluating or cloth_evaluating or squad_evaluating


def log_ops_count_enabled() -> bool:
    LOG_OPS_COUNT = os.getenv('LOG_OPS_COUNT', False)
    return LOG_OPS_COUNT


def log_row_ratio_enabled() -> bool:
    LOG_ROW_RATIO = os.getenv('LOG_ROW_RATIO', False)
    return LOG_ROW_RATIO


if __name__ == "__main__":
    if log_lb_enabled() and is_in_eval():
        print("hello world")
    if log_ops_count_enabled() and is_in_eval():
        print("你好世界")
