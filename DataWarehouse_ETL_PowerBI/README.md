### 🚌 Warsaw Public Transport: Data Warehouse & Delay Tracking System
This project involved building a data warehouse and reporting system to monitor public bus transport delays in Warsaw using official APIs from WTP (Warszawski Transport Publiczny).

🔄 ETL & Data Pipeline
- Implemented an ETL process using SSIS and Python scripts.

Combined:  

- A static table containing the list of all bus stops.

- Static API data (e.g., bus lines, scheduled routes).

- Dynamic API data — collected every minute — tracking all active buses, including real-time locations and timestamps.

- All data was transformed and loaded into a centralized data warehouse.

⭐ Data Modeling:  
- Structured the warehouse using a star schema to optimize Power BI performance

- Fact table: Captured the event of a bus approaching or arriving at the bus stop, enriched with geolocation and temporal data.

- Dimension tables: Stops, bus lines, timestamps, route metadata.

📊 Power BI Dashboard
Designed an interactive dashboard that:

- Visualized delays between planned and actual arrivals.

- Allowed users to select any bus stop and view upcoming buses with both scheduled and real-time arrival times.

- Enabled historical analysis of delay patterns across different routes and times.

📈 Future Potential:  
- While the dashboard used historical data, it was architected for real-time deployment.
- With minimal effort, statistical modeling can be layered to predict bus arrival times, potentially powering digital displays at bus stops city-wide.