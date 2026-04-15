Bangkok Airbnb Price Benchmarking
A data pipeline and machine learning project that identifies mispriced Airbnb listings in Bangkok by comparing actual prices against market-predicted prices.
Business Question
Are Bangkok Airbnb hosts pricing their listings correctly? Which neighbourhoods have the most mispriced listings?
What This Project Does

Cleans a raw 79-column, 28,000+ listing dataset (currency strings, percentage strings, boolean encoding, outlier removal)
Engineers features including amenities count and host response time encoding
Trains a Random Forest model (R² = 0.49) to predict fair market price based on room characteristics and neighbourhood
Flags each listing as Overpriced, Underpriced, or Fair (±20% threshold)
Identifies the top 10 most overpriced neighbourhoods in Bangkok

Key Findings

Most Bangkok listings are fairly priced, but more hosts underprice than overprice
Bang Kho Laen listings are overpriced by 191% on average
The model predicts budget listings well but struggles with premium properties — reflecting that luxury pricing depends on intangible factors not captured in the data

Tech Stack
Python, pandas, scikit-learn, matplotlib
Dataset
Inside Airbnb — Bangkok
