# EVE IskWorks Station Trader
This is a command line initiated gui program for market analysis of all major tradehubs of EVE. Its designed as a free to use service for aspiring Station Traders and Industrialists to fetch real time market data enabling you to find opportunities to buy and sell for a profit. The application showcases volume per day, ROI, potential profit as well as deviation vs Average price along with more.

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/85f7b4cc-2b09-4a4c-a260-ef288bdd80ca" />

# Installation Guide and Prerequisites
This application requires python 3.6+, although for ideal experience python 3.8+ is recommended, you can run it directly on Command Promt however an IDE makes it easier
1. Clone the RepositoryClone the project from the official GitHub repository to your local machine.
git clone [https://github.com/eveiskworks/EVE-IskWorks-Station-Trader.git](https://github.com/eveiskworks/EVE-IskWorks-Station-Trader.git)
cd EVE-IskWorks-Station-Trader
2. Install the required Python packages. This project primarily relies on requests for making HTTP calls to the ESI API and pandas for high-performance data manipulation and analysis.
3. pip install -r requirements.txt 
# If requirements.txt is missing or does not woek, install the core libraries manually:
pip install requests pandas numpy

<img width="752" height="85" alt="image" src="https://github.com/user-attachments/assets/399b5f52-0bfe-430c-bdf4-5b9bcdbdf174" />

# Running the Application
The applciation can be run in two ways directly from your command line or from an IDE, I recommend using an IDE especially if you are unfamiliar with programming.
Running the Application in an IDE is extremely simple, just open the folder where the program is located and click the run button

<img width="1918" height="746" alt="image" src="https://github.com/user-attachments/assets/248b56eb-42a7-4569-889f-f5a3afd5301d" />

Running this on the command line too isn't very complicated and can be easily done by opening the folder in command prompt using the following commands depending on your OS
python Eve_IskWorks_StationTrader.py
python3 Eve_IskWorks_StationTrader.py

<img width="1473" height="28" alt="image" src="https://github.com/user-attachments/assets/174602e8-fcd0-4db5-a7aa-8809e5ef1096" />

# Initial Setup/Repair

<img width="1931" height="1078" alt="image" src="https://github.com/user-attachments/assets/2a725670-033d-44ca-8995-ef7901051653" />

To Initially Setup the application during first time run or After an Update click on Update Static Data (Blue Circled). This will download all item names and their IDs from CCP for further use within the application, there are around 50k+ so it may take a minute. However it does save this file so it will only be needed during the first run (and future updates!)

After this select which tradehub your intrested in we currently support Jita, Amarr, Dodixie and Hek (Red Circled)

<img width="268" height="124" alt="image" src="https://github.com/user-attachments/assets/7a93ef4e-f55b-48f1-84e0-8b93eaea680f" />

Next apply whichever filters you are intrested in, We support a wide range of filters so that you can customize the data to however you want (Green Circled)

Lastly Click on Update Market Data (Pink Circled), this will check if existing data is cached, if not Market Data is downloaded from CCP. This market data is considered valid for 12 hours after which it is discarded and redownloaded. Volume is also downloaded for all items that match filters, since volume download is very slow it is considered valid for 7 days, however the top 100 performers have their volume updated every 12 hours. This is done mainly to reduce loading time and save bandwidth.

If you change any of the Filters click on Apply Filters so as to recalculate the data. (Brown Circle)

<img width="1882" height="88" alt="image" src="https://github.com/user-attachments/assets/7e897e90-d301-4047-80e9-93eb82e240ad" />

Clicking any of the headings (Purple Circled) sorts the data with respect to that heading, first in Decending and the Ascending Order

If you have any questions or doubts feel free to contact me on Discord
https://discord.gg/vAKd2zQpsU

If you liked this Project consider sending isk to EVE IskWorks to fund futre Development

This project is licensed under the Apache License. See the LICENSE.md file for full details.
