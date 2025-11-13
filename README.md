EVE IskWorks: Market Analysis ToolEVE IskWorks is a command-line utility designed to streamline market analysis for pilots in EVE Online. By leveraging ESI (EVE Swagger Interface) market endpoints, this tool fetches real-time order data and historical price averages across key regions to help you identify profitable trade routes, manufacturing opportunities, and market anomalies.✨ FeaturesFetches current buy/sell orders from major trade hubs (e.g., Jita, Amarr).Calculates profit margins for item movement across defined regions.Provides historical volume and price analysis for long-term tracking.Exports analysis data to a clean CSV or JSON format for external use.🛠️ Installation and SetupPrerequisitesThis application requires Python 3.8+ to run.You will also need to install the necessary Python libraries using pip.1. Clone the RepositoryFirst, clone the project from GitHub to your local machine:git clone [https://github.com/YourUsername/eve-iskworks.git](https://github.com/YourUsername/eve-iskworks.git)
cd eve-iskworks
2. Install DependenciesInstall the required packages. We use the standard requests for API calls and pandas for efficient data analysis.pip install -r requirements.txt 
# If you don't have a requirements.txt, use:
# pip install requests pandas EsiPy
3. Configuration (API Keys & IDs)The application requires specific EVE Item IDs (type_id) and Region IDs (region_id) to function.Create a Configuration File: Create a file named config.json in the project root directory.Populate config.json: Add your key settings here. You can look up type_id (Item ID) and region_id on sites like Fuzzworks.{
    "TARGET_REGIONS": [
        10000002,   // The Forge (Jita)
        10000043    // Domain (Amarr)
    ],
    "ITEM_IDS_TO_TRACK": [
        34,     // Tritanium
        35,     // Pyerite
        11396   // Warp Scrambler I
    ],
    "EXPORT_FORMAT": "csv"
}
🚀 How to UseOnce configured, the main script is run via the command line.1. Run the Market FetcherExecute the script to fetch the latest market data based on your config.json:python main_market_analysis.py
2. View OutputThe analysis will be saved to a folder named output/ in the project root.Console Output: A summary of the top 5 highest profit margin trades will be displayed in your terminal.File Output: The full dataset will be written to a file based on your configuration (e.g., output/market_analysis_2025-11-13.csv).🖼️ Example OutputHere is an example of the kind of data visualization this tool can help generate:📄 LicenseThis project is licensed under the MIT License - see the LICENSE.md file for details.
