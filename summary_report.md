# Summary Report - Credit Card Customer Segmentation

## Business problem and dataset
The credit cards division of a large Indian private bank wants to stop using one-size-fits-all offers for card upgrades, rewards and credit-limit revisions. My task was to group existing cardholders into behavioural segments so that Relationship Managers can make the right offer to the right customer. I used the Kaggle *Credit Card Dataset for Clustering* (`CC_GENERAL.csv`): 8,950 cardholders and 18 columns, of which 17 numeric columns were used after dropping `CUST_ID`.

## Feature engineering and preprocessing
Only `CREDIT_LIMIT` (1 value) and `MINIMUM_PAYMENTS` (313 values) had missing data, so I filled both with the median because they are right-skewed. I created four behavioural features: `Monthly_Avg_Purchase`, `Monthly_Avg_Cash_Advance`, `Limit_Usage` and `Payment_to_Minpayment_Ratio`. The monetary columns had very long right tails, so I capped extreme values above Q3 + 3 x IQR (1,483 values, no rows dropped), applied `log1p` to 11 monetary columns, and then used `StandardScaler` on all 21 features, since K-Means, Ward linkage and DBSCAN all depend on distance. PCA needed 10 components for 90% variance, which is not a big reduction, so I clustered on the full scaled data.

## Algorithm results
| Algorithm | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | Noise |
|---|---|---|---|---|---|
| K-Means (k=4) | 4 | 0.196 | 1.71 | 1,987 | 0% |
| Agglomerative (Ward) | 4 | 0.138 | 1.98 | 1,422 | 0% |
| DBSCAN (eps=1.5, min_samples=10) | 8 | -0.148 | 1.23 | 35 | 28.4% |

K-Means performed best by internal metrics and was very stable (silhouette 0.1959 +/- 0.00007 over five seeds). This matched my business intuition, because it gave four clear and explainable personas. Ward gave almost the same groups (about 74% of customers land in the matching cluster). DBSCAN did not fit this data: it either marked almost everyone as noise or gave one giant cluster with a few tiny ones. Its low Davies-Bouldin value is only because noise points are excluded. Overall silhouette values are low (about 0.2), so segment borders are soft.

## Cardholder segments (K-Means)
- **Premium Transactors (1,655):** highest limit and spending, buy almost every month, many pay in full, little cash advance.
- **Revolvers (2,325):** highest balance and minimum payments, 67% limit usage, only 2% pay in full; they bring interest income.
- **Cash-Advance Reliant (2,319):** almost no purchases but the highest cash advance, so the card works like a loan.
- **Low-Activity Cardholders (2,651):** lowest balance, limit and payments, only 7% limit usage.

DBSCAN's noise points (heavy spenders and heavy cash-advance users) should go to manual Relationship Manager and credit-risk review.

## Next steps
1. Add richer data: income, credit bureau score, merchant categories, delinquency history and a longer monthly time series.
2. Link segments to attrition and delinquency labels and use semi-supervised refinement of the borders.
3. Wrap the saved scaler and model in a real-time scoring API so a new cardholder is assigned a segment automatically.
