"""
Main entry point for the Hybrid Recommendation System.
"""

import argparse
import sys
import os
import subprocess
import webbrowser
import time
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.interface import main as cli_main

def launch_streamlit():
    """Launch Streamlit web interface automatically."""
    print("🚀 Launching Streamlit Web Interface...")
    print("📱 The web interface will open automatically in your browser.")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("-" * 60)
    
    # Launch Streamlit
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "src/web_interface.py",
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    # Start Streamlit in a separate process
    process = subprocess.Popen(
        streamlit_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a moment for Streamlit to start
    time.sleep(3)
    
    # Open browser automatically
    try:
        webbrowser.open('http://localhost:8501')
        print("✅ Web interface opened in your default browser!")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("📖 Please manually open: http://localhost:8501")
    
    return process

def main():
    """Main function to run the recommendation system."""
    parser = argparse.ArgumentParser(
        description='Hybrid Recommendation System by DevanshSrajput',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with web interface (DEFAULT)
  python main.py
  
  # Load data and train models via CLI
  python main.py --cli --load-data 100k --train
  
  # Get recommendations for user 1 via CLI
  python main.py --cli --user-id 1 --method hybrid --top-k 10
  
  # Start interactive CLI mode
  python main.py --cli --interactive
  
  # Run evaluation via CLI
  python main.py --cli --load-data 100k --train --evaluate
  
  # Force web interface launch
  python main.py --web
        """
    )
    
    # Add web interface options
    parser.add_argument('--web', action='store_true', default=False,
                       help='Launch web interface (Streamlit)')
    parser.add_argument('--cli', action='store_true', default=False,
                       help='Use command line interface instead of web')
    parser.add_argument('--no-browser', action='store_true', default=False,
                       help='Don\'t open browser automatically (web mode only)')
    
    # CLI options
    parser.add_argument('--load-data', type=str, default='100k',
                       help='Load and preprocess data (100k, 1m, 10m)')
    parser.add_argument('--train', action='store_true',
                       help='Train all models')
    parser.add_argument('--user-id', type=int,
                       help='User ID for recommendations')
    parser.add_argument('--method', type=str, default='hybrid',
                       choices=['collaborative', 'content_based', 'hybrid'],
                       help='Recommendation method')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of recommendations')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate models')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Print welcome message
    print("=" * 70)
    print("🎬 HYBRID RECOMMENDATION SYSTEM")
    print(f"👤 Developer: DevanshSrajput")
    print(f"📅 Last Updated: 2025-06-17")
    print("=" * 70)
    
    # Determine mode
    if args.cli or any([args.load_data and args.load_data != '100k', args.train, 
                       args.user_id is not None, args.evaluate, args.interactive]):
        # CLI mode
        print("🖥️  Starting Command Line Interface...")
        cli_main()
    else:
        # Default to web interface
        print("🌐 Starting Web Interface Mode...")
        
        # Check if Streamlit is installed
        try:
            import streamlit
            print("✅ Streamlit found!")
        except ImportError:
            print("❌ Error: Streamlit is not installed.")
            print("📦 Please install it with: pip install streamlit")
            print("🔄 Or run: pip install -r requirements.txt")
            return
        
        # Launch Streamlit
        try:
            if not args.no_browser:
                process = launch_streamlit()
            else:
                # Launch without opening browser
                streamlit_cmd = [
                    sys.executable, "-m", "streamlit", "run", 
                    "src/web_interface.py",
                    "--server.port=8501"
                ]
                process = subprocess.Popen(streamlit_cmd)
                print("🌐 Streamlit started at: http://localhost:8501")
            
            print("\n🎯 Web Interface Features:")
            print("   • Interactive data exploration")
            print("   • Real-time recommendations")
            print("   • Model comparison and evaluation")
            print("   • Beautiful visualizations")
            print("   • User-friendly interface")
            print("\n⏹️  Press Ctrl+C to stop the server")
            
            # Keep the main process alive
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                process.terminate()
                process.wait()
                print("✅ Goodbye!")
                
        except Exception as e:
            print(f"❌ Error launching web interface: {e}")
            print("🔄 Falling back to CLI mode...")
            cli_main()

if __name__ == "__main__":
    main()