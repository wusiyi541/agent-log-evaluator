import sys
import os
from datetime import datetime

# 确保在导入任何模块之前加载 .env 文件
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    load_config,
    run_evaluation,
    list_test_cases,
    list_log_turns,
)


def main():
    print("=" * 60)
    print("📊 日志评测系统 - 简易入口")
    print("=" * 60)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="日志评测系统")
    parser.add_argument("--mode", choices=["evaluate", "parse", "list-tests", "list-logs", "report"], 
                       default="evaluate", help="运行模式")
    parser.add_argument("--log", help="日志文件路径")
    parser.add_argument("--test-set", help="测试集文件路径")
    parser.add_argument("--mapping", help="手动映射文件路径")
    parser.add_argument("--output", help="输出目录")
    parser.add_argument("--dates", help="日期过滤")
    parser.add_argument("--keywords", help="关键词过滤")
    parser.add_argument("--test-range", help="测试范围 (如: 1-10)")
    parser.add_argument("--report", help="JSON报告文件路径 (用于生成HTML)")
    
    args = parser.parse_args()
    
    config = load_config(
        log_file=args.log,
        test_set_file=args.test_set,
        mapping_file=args.mapping,
        output_dir=args.output,
        dates=args.dates,
        keywords=args.keywords,
    )
    
    if args.mode == "list-tests":
        list_test_cases(config)
        return
    
    if args.mode == "list-logs":
        list_log_turns(config)
        return
    
    if args.mode == "report":
        if not args.report:
            print("[ERROR] 请使用 --report 指定JSON报告文件")
            sys.exit(1)
        
        from report_generator import generate_report_from_json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = f"政务数据问答助手测评报告_{timestamp}.html"
        generate_report_from_json(args.report, os.path.join(config.output.output_dir, html_file))
        return
    
    if args.mode == "parse":
        result = run_evaluation(config)
        print("\n[INFO] 解析完成，跳过评测")
        return
    
    result = run_evaluation(
        config,
        turn_range=None,
        test_range=args.test_range,
    )
    
    if result:
        from report_generator import generate_html_report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = os.path.join(config.output.output_dir, f"政务数据问答助手测评报告_{timestamp}.html")
        generate_html_report(result, html_file)
        print(f"\n🎉 评测完成！请查看报告: {html_file}")


if __name__ == "__main__":
    main()
