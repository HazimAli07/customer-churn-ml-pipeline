# Dataset

This project uses IBM's **Telco Customer Churn** sample dataset. It describes
7,043 fictional telecom customers and whether each customer left during the
last month.

## Source

- Dataset: [`Telco-Customer-Churn.csv`](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)
- Original project: [IBM telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- Original repository license: [Apache License 2.0](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE)

The source repository was archived by IBM in July 2024 and remains publicly
available in read-only form.

## Fields used

The target is `Churn` (`Yes` or `No`). Model inputs cover:

- Customer profile: gender, senior status, partner and dependent status
- Account history: tenure, contract, billing and payment method
- Services: phone, internet, security, backup, support and streaming
- Charges: monthly and total charges

`customerID` is excluded because it identifies a customer rather than
describing behaviour. Blank values in `TotalCharges` are converted to missing
numeric values and imputed within the training pipeline.
