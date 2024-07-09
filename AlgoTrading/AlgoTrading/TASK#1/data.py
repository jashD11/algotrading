import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def RSI(df):
    "returns a pandas series which represents the RSI"
    window = 14
    # Calculate daily price changes
    delta = df['Close'].diff()
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)#where delta>0:gain = 0 where delta <=0
    loss = -delta.where(delta < 0, 0)
    # Calculate the average gain and average loss
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    # Calculate Relative Strength (RS)
    RS = avg_gain / avg_loss
    # Calculate RSI
    RSI = 100 - (100 / (1 + RS))
    return RSI

def movingAverage(df , wind):
    return df['Close'].rolling(window=wind).mean()#creating a new pandas sequence with a rolling window of a particular length and taking its mean



df = pd.read_csv("BRITANNIA.NS.csv", index_col=0)
#CLEANING DATA
df.dropna(inplace = True)#removes 03/15
#print(df.duplicated()) checked if there are any duplicaetd ones by using this function
#checked if any dates are wrong:none
#DONE WITH PRE PROCESSING

#RSI and MACD
#d1 = 13
#d2 = 26
#df[f'{d1}-day MA'] = movingAverage(df,d1)
#df[f'{d2}-day MA'] = movingAverage(df,d2)
#df['MACD'] = df[f'{d1}-day MA'] - df[f'{d2}-day MA']
#df['RSI'] = RSI(df)
#plt.plot(df.index,df['RSI'],label = 'RSI')
#plt.plot(df.index,df['MACD'],label = 'MACDC')
#plt.xlabel('Date')
#plt.ylabel('RSI')
#plt.title('RSI and MACD')

#MOVING AVERAGE STARTEGY IMPLEMETATION
signal = pd.DataFrame(index=df['Close'].index)
signal['Close'] = df['Close']
signal['50DMA'] = movingAverage(df,50)
signal['20DMA'] = movingAverage(df,20)
signal['signal'] = 0
signal['signal'][50:] = np.where(signal['50DMA'][50:]>signal['20DMA'][50:],0,1)# 3 arguments in np.where-condition,true,false
signal['call'] = signal['signal'].diff()#tells about the buy or sell call
signal['return'] = signal['signal']*signal['Close'].pct_change().shift(1)
capital = 100000
signal['equity'] = capital * (1 + signal['return']).cumprod()#calculates the cumalative product...the other option i using a loop to calculate eequity based on previous value
print(signal.to_string())
plt.plot(signal.index , signal['equity'])
#plotting 20day and 50 day moving average
#plt.plot(df.index, df['Close'], label='Close Price')
#plt.plot(df.index, df['20DMA'], label='20-day MA')
#plt.plot(df.index, df['50DMA'], label='50-day MA')
plt.xlabel('Date')
plt.ylabel('equity')
#plt.title('Close Price and Moving Averages')
plt.legend()
plt.show()
#print(df.to_string())