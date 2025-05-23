# 🚍 Warsaw Public Transport: Data Engineering & Analytics Project

A comprehensive data warehouse and analytics solution for tracking and understanding bus delays in Warsaw—built to showcase advanced data engineering, ETL, and dashboarding expertise.

---

## 🏆 Project Highlights

- **End-to-End Data Pipeline:** Designed and implemented an automated ETL workflow, integrating Python and SSIS, to collect, transform, and unify both static and real-time data from the official Warszawski Transport Publiczny (WTP) APIs.
- **Modern Data Modeling:** Developed a robust star schema data warehouse, engineered for fast, flexible analytics and seamless Power BI integration.
- **Actionable Analytics:** Created a Power BI dashboard that provides actionable insights into bus delays, routes, and city-wide historical trends—empowering transit authorities and commuters alike.
- **Scalable Design:** The architecture supports real-time analytics and is ready for predictive modeling, enabling future deployment at city scale.

---

## 🔄 Data Pipeline & ETL

- **Sources:**
  - **Static Reference Data:** Bus stops, bus lines, and scheduled routes.
  - **Real-Time Data:** Live positions and timestamps of all active buses, collected every minute.

- **ETL Process:**
  - Automated collection and cleansing of both static and dynamic data.
  - Transformation and integration into a centralized data warehouse.
  - Scheduled data loads to ensure up-to-date analytics.

---

## ⭐ Data Warehouse Schema

- **Star Schema Model:**
  - **Fact Table:** Each record captures a bus event (arrival/approach), including timestamp and geolocation.
  - **Dimension Tables:** Bus stops, bus lines, route metadata, and time.
- **Performance:** Optimized for Power BI, enabling fast slice-and-dice analysis across millions of records.

---

## 📊 Power BI Dashboard

- **Interactive Features:**
  - Visualizes delays between scheduled and actual arrivals for any bus stop.
  - Allows users to explore upcoming arrivals, route performance, and historical delay patterns.
  - Drill-down capabilities for temporal and spatial analyses.

- **Sample Insights:**
  - Identify chronically delayed routes or stops.
  - Analyze delay patterns by time of day, day of week, or specific lines.
  - Support for “next bus” real-time queries.

---

## 🚀 Future Directions

- **Predictive Analytics:** The system is architected for easy integration of statistical or machine learning models to forecast bus arrival times—paving the way for city-wide real-time displays and smarter urban mobility.
- **Scalability:** Minimal adjustments needed to support continuous data streaming and predictive deployments.

---

## 🛠️ Technologies Used

- **ETL:** Python, SSIS
- **Data Warehouse:** SQL (star schema modeling)
- **Analytics & Visualization:** Power BI

---
