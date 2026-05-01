"""
Master Pipeline Runner — Pharma Forecasting Platform
======================================================
Runs the entire pipeline in correct order with one command.

Usage: python run_all.py
"""

import subprocess
import sys
import time
from datetime import datetime

# Color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    print(f"""
{Colors.BLUE}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     💊  INDIA PHARMA FORECASTING PLATFORM  💊               ║
║                                                              ║
║     Multi-Source Real Data Intelligence System              ║
║     Sources: IHME • ICMR • NFHS-5 • UN • data.gov.in       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}
    """)


def run_script(script_name, description, optional=False):
    """Run a Python script and report results"""
    print(f"\n{Colors.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}")
    print(f"{Colors.BOLD}🚀 STEP: {description}{Colors.END}")
    print(f"   Running: {script_name}")
    print(f"{Colors.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}")
    
    start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ SUCCESS — {description} completed in {elapsed:.1f}s{Colors.END}")
            return True
        else:
            if optional:
                print(f"{Colors.YELLOW}⚠ SKIPPED — {description} (optional){Colors.END}")
                return True
            else:
                print(f"{Colors.RED}❌ FAILED — {description}{Colors.END}")
                return False
                
    except FileNotFoundError:
        if optional:
            print(f"{Colors.YELLOW}⚠ SKIPPED — {script_name} not found (optional){Colors.END}")
            return True
        else:
            print(f"{Colors.RED}❌ FAILED — {script_name} not found{Colors.END}")
            return False
    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}❌ TIMEOUT — {script_name} took too long{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ ERROR — {e}{Colors.END}")
        return False


def main():
    print_banner()
    
    start_time = datetime.now()
    print(f"{Colors.BOLD}🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    
    # Pipeline steps in execution order
    pipeline = [
        # (script, description, optional)
        ("process_real_data.py", "Process Real Data (IHME, NFHS, ICMR)", False),
        ("forecasting.py", "Main Forecasting Pipeline (10K Monte Carlo + ARIMA + ML)", False),
        ("process_state_data.py", "State-Level Data Integration (31 states)", False),
        ("state_forecasting.py", "State × Year Forecasting (620 rows)", False),
        ("sensitivity_analysis.py", "Sensitivity Analysis (Tornado Charts)", True),
        ("bayesian_forecasting.py", "Bayesian Hierarchical Model", True),
        ("live_data_pipeline.py", "Live API Data Refresh", True),
    ]
    
    results = []
    
    for script, description, optional in pipeline:
        success = run_script(script, description, optional)
        results.append((script, success))
    
    # Final summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n{Colors.BOLD}{'='*64}{Colors.END}")
    print(f"{Colors.BOLD}📊 PIPELINE EXECUTION SUMMARY{Colors.END}")
    print(f"{'='*64}")
    
    success_count = sum(1 for _, s in results if s)
    total = len(results)
    
    for script, success in results:
        icon = "✅" if success else "❌"
        color = Colors.GREEN if success else Colors.RED
        print(f"  {color}{icon} {script}{Colors.END}")
    
    print(f"\n{Colors.BOLD}🎯 Result: {success_count}/{total} successful{Colors.END}")
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print(f"🕐 Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 PIPELINE COMPLETED SUCCESSFULLY!{Colors.END}")
        print(f"\n{Colors.YELLOW}📊 Next Steps:{Colors.END}")
        print(f"   1. Open Power BI: outputs/forecast_combined_state_year.csv")
        print(f"   2. Launch web dashboard: {Colors.BOLD}streamlit run app.py{Colors.END}")
        print(f"   3. Check outputs/ folder for generated visualizations")
    else:
        print(f"\n{Colors.RED}⚠ Some steps failed. Check errors above.{Colors.END}")
    
    print(f"\n{Colors.BLUE}{'='*64}{Colors.END}\n")


if __name__ == "__main__":
    main()