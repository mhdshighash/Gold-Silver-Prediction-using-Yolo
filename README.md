# Gold-Silver-Prediction 
It Forecast Gold and Silver price rate for 7 days.  
The Price values is in dollars.  
It will calculate 1 Ounce of Gold & Silver.     

**---Requirements---**    
# Gold-Prdeiction
import yfinance as yf  
import pandas as pd  
from sklearn.ensemble import RandomForestRegressor  
from datetime import datetime, timedelta  


GRAMS_PER_TROY_OUNCE = 31.1034768  
GST_RATE = 0.03  

# Silver-Prediction
import yfinance as yf  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.ensemble import RandomForestRegressor  
from datetime import datetime, timedelta  
