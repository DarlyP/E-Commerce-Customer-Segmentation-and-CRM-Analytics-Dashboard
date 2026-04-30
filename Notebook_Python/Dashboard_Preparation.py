import marimo

__generated_with = "0.23.4"
app = marimo.App(auto_download=["html", "ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **E-Commerce Customer Segmentation and CRM Analytics Dashboard - Dashboard Preparation**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **The aim of this notebook:**
    > - Build a final customer-level dataset that is ready for Power BI dashboard development
    > - Combine customer value, behavior, engagement, loyalty, revenue quality, and retention risk indicators into one structured analytical table
    > - Create business-friendly customer segments that support segmentation, retention, and prioritization analysis
    > - Identify customers with stable behavior, revenue leakage signals, promotion dependency, and elevated retention risk
    > - Generate supporting dimension tables for consistent sorting, filtering, and interpretation in Power BI
    > - Prepare clean export files that can be directly used for dashboard modeling and business reporting

    > **Expected Output:**
    > - A final `fact_customer_dashboard` table at customer level with key metrics, segment labels, risk flags, business priority, and recommended actions
    > - Customer segmentation outputs such as FM segment, behavioral segment, revenue quality label, and retention risk flag
    > - Retention risk scoring that classifies customers into Stable, Watchlist, At Risk, and Critical Risk groups
    > - Business priority classification that helps translate customer behavior into practical action groups
    > - Dimension tables for retention risk, behavior segment, and business priority to support Power BI relationships and visual sorting
    > - Export-ready CSV files for Power BI dashboard pages such as Executive Overview, Segmentation, Revenue Quality, Retention Risk, Stable vs Risky, and Business Priority

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Import Library**

    ---
    """)
    return


@app.cell
def _():
    # Import necessary libraries
    import numpy as np
    import pandas as pd

    # Importing MinMaxScaler from sklearn.preprocessing
    from sklearn.preprocessing import MinMaxScaler


    return MinMaxScaler, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## **Data Loading & Preparation**

    ---
    """)
    return


@app.cell
def _(pd):
    # Load the dataset
    df = pd.read_csv(r"C:\Users\user\Documents\Coding\Portofolio\E-Commerce-Customer-Segmentation-and-CRM-Analytics-Dashboard\Dataset\customer_data_full.csv")
    df.head()
    return (df,)


@app.cell
def _(df):
    # Print the shape of the dataset to understand its dimensions
    print("Dataset shape:", df.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Data loaded succesfully

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Helper Functions**

    ---
    """)
    return


@app.cell
def _(MinMaxScaler, np, pd):
    # Define utility functions for data validation and transformation
    def require_columns(df: pd.DataFrame, required_columns: list[str], step_name: str = "validation") -> None:
        """Raise an error if required columns are not available in the dataset."""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"[{step_name}] The following columns are missing from the dataset: {missing}"
            )

    # Define a function to normalize binary columns
    def normalize_binary(series: pd.Series) -> pd.Series:
        """Convert yes/no, true/false, or 0/1 values into numeric 0/1 values."""
        s_num = pd.to_numeric(series, errors="coerce")

        if s_num.dropna().isin([0, 1]).all():
            return s_num.astype("Int64")

        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "y": 1,
                "n": 0,
            })
            .astype("Int64")
        )

    # Define a function to cap outliers using quantile thresholds
    def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """Cap outliers using lower and upper quantile thresholds."""
        low = s.quantile(lower)
        high = s.quantile(upper)
        return s.clip(lower=low, upper=high)

    # Define a function to scale numeric columns to the 0-1 range using MinMax scaling
    def minmax_series(series: pd.Series) -> pd.Series:
        """Scale a single numeric column to the 0-1 range."""
        scaler = MinMaxScaler()
        values = scaler.fit_transform(series.to_frame()).flatten()
        return pd.Series(values, index=series.index)

    # Define a function to calculate a weighted score after normalizing selected columns
    def weighted_score(df: pd.DataFrame, cols: list[str], weights: list[float]) -> np.ndarray:
        """Calculate a weighted score after normalizing all selected columns with MinMax scaling."""
        require_columns(df, cols, step_name="weighted_score")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Total weight must be 1. Current total: {sum(weights)}")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[cols])
        return np.dot(scaled, np.array(weights))

    # Define a function to create quantile-based labels using ranking to handle duplicate values safely
    def qlabel(series: pd.Series, labels=("Low", "Medium", "High")) -> pd.Series:
        """Create quantile-based labels using ranking to handle duplicate values safely."""
        return pd.qcut(
            series.rank(method="first"),
            q=len(labels),
            labels=labels
        )

    return (
        normalize_binary,
        qlabel,
        require_columns,
        weighted_score,
        winsorize_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This section contains small reusable helper functions used throughout the notebook:

    * Column validation
    * Binary value normalization
    * Winsorization
    * Weighted score calculation
    * Quantile-based labeling

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Select Analysis Column**

    ---
    """)
    return


@app.cell
def _(df, require_columns):
    # Select relevant columns for analysis based on domain knowledge and EDA insights
    analysis_columns = [
        # Identifier
        "user_id",

        # Value / revenue
        "monthly_spend",
        "weekly_purchases",
        "average_order_value",
        "lifetime_value_proxy",
        "risk_adjusted_value",
        "spend_per_purchase",
        "revenue_proxy",
        "adjusted_revenue",
        "revenue_efficiency",

        # Engagement / behavior
        "daily_session_time_minutes",
        "product_views_per_day",
        "app_usage_frequency",
        "wishlist_items_count",
        "notification_response_rate",
        "engagement_score",
        "browse_to_buy_ratio",

        # Leakage / risk
        "return_rate",
        "return_frequency",
        "cart_abandonment_rate",
        "checkout_abandonments_per_month",
        "risk_score",
        "coupon_usage_frequency",
        "coupon_dependency",
        "churn_proxy",
        "stability_score",
        "retention_strength",
        "risk_exposure",

        # Loyalty
        "loyalty_program_member",
        "premium_subscription",
        "referral_count",
        "brand_loyalty_score",

        # Customer profile
        "age",
        "gender",
        "country",
        "urban_rural",
        "income_level",
        "employment_status",
        "education_level",
        "household_size",
    ]

    require_columns(df, analysis_columns, step_name="initial column selection")

    df_segment = df[analysis_columns].copy()

    print("Selected shape:", df_segment.shape)
    df_segment.head()
    return (df_segment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This list combines the requirements of:
    - final Power BI schema
    - scoring from the Behavioral notebook
    - customer labels from the Shopper notebook

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Data Type Preparation**

    ---
    """)
    return


@app.cell
def _(df_segment, normalize_binary, pd):
    # Convert numeric columns to appropriate data types and normalize binary columns
    numeric_cols = [
        "monthly_spend",
        "weekly_purchases",
        "average_order_value",
        "lifetime_value_proxy",
        "risk_adjusted_value",
        "spend_per_purchase",
        "revenue_proxy",
        "adjusted_revenue",
        "revenue_efficiency",
        "daily_session_time_minutes",
        "product_views_per_day",
        "app_usage_frequency",
        "wishlist_items_count",
        "notification_response_rate",
        "engagement_score",
        "browse_to_buy_ratio",
        "return_rate",
        "return_frequency",
        "cart_abandonment_rate",
        "checkout_abandonments_per_month",
        "risk_score",
        "coupon_usage_frequency",
        "coupon_dependency",
        "churn_proxy",
        "stability_score",
        "retention_strength",
        "risk_exposure",
        "referral_count",
        "brand_loyalty_score",
        "age",
        "household_size",
    ]

    # Binary columns that need normalization to 0/1
    binary_cols = [
        "loyalty_program_member",
        "premium_subscription",
    ]

    # Convert numeric columns to numeric data types, coercing errors to NaN
    for num_col in numeric_cols:
        df_segment[num_col] = pd.to_numeric(df_segment[num_col], errors="coerce")

    # Normalize binary columns to 0/1
    for bin_col in binary_cols:
        df_segment[bin_col] = normalize_binary(df_segment[bin_col])

    # Check for missing values after type conversion and normalization
    missing_summary = (
        df_segment
        .isna()
        .sum()
        .sort_values(ascending=False)
        .to_frame("missing_count")
    )
    return (missing_summary,)


@app.cell
def _(df_segment, missing_summary):
    # Calculate missing percentage for each column 
    missing_summary["missing_percentage"] = (missing_summary["missing_count"] / len(df_segment)) * 100
    missing_summary.head(20)
    return


@app.cell
def _(df_segment):
    # Data types after conversion and normalization
    df_segment.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Numeric columns are converted to numeric. Binary columns such as loyalty and premium are normalized to 0/1. No missing values ​​were found, allowing for further analysis.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Optional Shopper Labels**

    ---
    """)
    return


@app.cell
def _(df_segment, np):
    # Create human-readable labels for loyalty program membership and premium subscription status
    df_segment["loyalty_program_member_label"] = df_segment["loyalty_program_member"].map({
        1: "Loyalty Member",
        0: "Non-Loyalty"
    })

    # Create human-readable labels for premium subscription status
    df_segment["premium_subscription_label"] = df_segment["premium_subscription"].map({
        1: "Premium",
        0: "Non-Premium"
    })

    # Create a combined customer group label based on loyalty program membership and premium subscription status
    df_segment["customer_group"] = np.select(
        [
            (df_segment["loyalty_program_member"] == 1) & (df_segment["premium_subscription"] == 1),
            (df_segment["loyalty_program_member"] == 1) & (df_segment["premium_subscription"] == 0),
            (df_segment["loyalty_program_member"] == 0) & (df_segment["premium_subscription"] == 1),
            (df_segment["loyalty_program_member"] == 0) & (df_segment["premium_subscription"] == 0),
        ],
        [
            "Loyalty + Premium",
            "Loyalty Only",
            "Premium Only",
            "Standard",
        ],
        default="Unknown"
    )
    return


@app.cell
def _(df_segment):
    # Display the new columns to verify correct labeling
    df_segment[
            [
                "loyalty_program_member",
                "premium_subscription",
                "loyalty_program_member_label",
                "premium_subscription_label",
                "customer_group",
            ]
        ].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This section uses logic from Shopper Behavior Analysis to create additional labels:
    - `loyalty_program_member_label`
    - `premium_subscription_label`
    - `customer_group`

    These columns are not required for the final schema, but are useful as additional slicers in Power BI.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Outlier Handling**

    ---
    """)
    return


@app.cell
def _(df_segment, winsorize_series):
    # Cap outliers in key numeric columns to reduce skewness and improve model performance
    winsor_cols = [
        "monthly_spend",
        "weekly_purchases",
        "lifetime_value_proxy",
        "risk_adjusted_value",
        "daily_session_time_minutes",
        "product_views_per_day",
        "revenue_proxy",
        "adjusted_revenue",
        "return_rate",
        "cart_abandonment_rate",
        "risk_score",
        "coupon_dependency",
        "risk_exposure",
    ]

    for win_col in winsor_cols:
        df_segment[win_col] = winsorize_series(df_segment[win_col], lower=0.01, upper=0.99)

    df_segment[winsor_cols].describe().T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    **Insight:**

    Capping is applied only to selected numeric columns that are important for customer segmentation, revenue quality analysis, and retention risk scoring. These variables are highly sensitive to extreme values because they directly influence customer value, behavioral intensity, leakage pressure, and risk classification.

    The purpose of winsorization is not to remove unusual customers from the dataset. Instead, it limits the influence of extreme values by capping them at the 1st and 99th percentile. This keeps all customer records in the analysis while reducing the risk that a small number of extreme observations will dominate the results.

    For example, variables such as `monthly_spend`, `lifetime_value_proxy`, `adjusted_revenue`, and `risk_adjusted_value` often have skewed distributions. A small group of very high-value customers can pull averages upward and make customer segments look more valuable than they actually are for the broader customer base.

    Engagement-related variables such as `daily_session_time_minutes` and `product_views_per_day` can also contain unusually high values. These may represent very active customers, but they may also come from tracking noise, repeated browsing behavior, or abnormal usage patterns. Capping helps keep these signals useful without allowing them to distort behavioral scoring.

    Risk and leakage variables such as `return_rate`, `cart_abandonment_rate`, `risk_score`, `coupon_dependency`, and `risk_exposure` are also capped because they are directly used in retention risk interpretation. If these variables contain extreme values, the final risk score may become too aggressive and classify customers as risky mainly because of a few abnormal observations.

    This selective approach is important. Not every numeric column needs to be winsorized. Applying capping too broadly can over-smooth the dataset and reduce meaningful customer differences. Therefore, only business-critical variables with strong outlier sensitivity are capped.

    After capping, the dataset becomes more stable for downstream processes such as weighted scoring, FM segmentation, behavioral segmentation, revenue quality analysis, and retention risk scoring. The result is a more robust analytical dataset that still preserves customer-level variation but is less affected by extreme outliers.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Business Feature Construction**

    ---
    """)
    return


@app.cell
def _(df_segment, weighted_score):
    # Calculate composite scores for value, behavior, leakage pressure, and loyalty strength using weighted averages of relevant features
    df_segment["value_core"] = weighted_score(
        df_segment,
        ["monthly_spend", "lifetime_value_proxy", "risk_adjusted_value", "adjusted_revenue"],
        [0.35, 0.25, 0.20, 0.20]
    )

    # Calculate behavior core score using weighted average of engagement and activity metrics
    df_segment["behavior_core"] = weighted_score(
        df_segment,
        [
            "daily_session_time_minutes",
            "product_views_per_day",
            "app_usage_frequency",
            "wishlist_items_count",
            "notification_response_rate",
            "engagement_score",
            "browse_to_buy_ratio",
        ],
        [0.20, 0.15, 0.15, 0.10, 0.15, 0.15, 0.10]
    )

    # Calculate leakage pressure score using weighted average of risk and abandonment metrics
    df_segment["leakage_pressure"] = weighted_score(
        df_segment,
        [
            "cart_abandonment_rate",
            "checkout_abandonments_per_month",
            "return_rate",
            "return_frequency",
        ],
        [0.35, 0.25, 0.20, 0.20]
    )

    # Calculate loyalty strength score using weighted average of loyalty program membership, premium subscription, referral count, and brand loyalty score
    df_segment["loyalty_strength"] = weighted_score(
        df_segment,
        [
            "loyalty_program_member",
            "premium_subscription",
            "referral_count",
            "brand_loyalty_score",
        ],
        [0.25, 0.20, 0.20, 0.35]
    )

    # Display the new composite scores to verify correct calculation
    df_segment[
            ["value_core", "behavior_core", "leakage_pressure", "loyalty_strength"]
        ].describe().T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This step creates four composite scores that summarize customer quality from different business perspectives:

    - `value_core`
    - `behavior_core`
    - `leakage_pressure`
    - `loyalty_strength`

    These scores are created because individual columns are useful, but they can be too detailed for business interpretation. A composite score helps combine several related variables into one interpretable metric, making it easier to compare customers, build segments, and support dashboard-level analysis.

    Before calculating the score, each selected feature is normalized using MinMax scaling. This is important because the input columns have different units and ranges. For example, `monthly_spend` may be measured in currency, `engagement_score` may already be between 0 and 1, while `referral_count` is a count variable. Scaling them first ensures that no variable dominates the score only because it has a larger numeric range.

    ---

    **1. Value Core**

    `value_core` represents the economic contribution of each customer.

    It combines:

    - `monthly_spend`
    - `lifetime_value_proxy`
    - `risk_adjusted_value`
    - `adjusted_revenue`

    These variables are selected because they describe both current and long-term customer value. `monthly_spend` receives the highest weight because it reflects the most direct and current revenue contribution. `lifetime_value_proxy` is included to capture longer-term commercial potential. `risk_adjusted_value` and `adjusted_revenue` are added to make the value score more realistic, because high revenue is not always healthy if it is exposed to risk or leakage.

    This score helps identify customers who are commercially important and should receive stronger attention in segmentation, retention, and business priority analysis.

    ---

    **2. Behavior Core**

    `behavior_core` represents the level and quality of customer activity.

    It combines:

    - `daily_session_time_minutes`
    - `product_views_per_day`
    - `app_usage_frequency`
    - `wishlist_items_count`
    - `notification_response_rate`
    - `engagement_score`
    - `browse_to_buy_ratio`

    These variables are selected because they describe how actively customers interact with the platform. Session time, product views, and app usage show activity intensity. Wishlist count reflects purchase interest. Notification response rate shows marketing responsiveness. Engagement score summarizes overall interaction quality, while browse-to-buy ratio helps capture whether browsing activity is connected to buying behavior.

    This score is useful because customer value should not only be understood from spending. A customer may not currently spend a lot but may show strong activity, high interest, and good engagement. Those customers may represent future growth potential.

    ---

    **3. Leakage Pressure**

    `leakage_pressure` represents the level of revenue loss or friction associated with each customer.

    It combines:

    - `cart_abandonment_rate`
    - `checkout_abandonments_per_month`
    - `return_rate`
    - `return_frequency`

    These variables are selected because they indicate where potential revenue is lost before or after purchase. Cart abandonment and checkout abandonment show pre-purchase friction, where customers show intent but do not complete the transaction. Return rate and return frequency show post-purchase leakage, where completed purchases may not fully translate into retained revenue.

    This score is important because a customer can generate high spending but still create weak revenue quality if they frequently abandon checkout, return products, or create unstable value. Higher leakage pressure means the customer may require funnel improvement, product fit analysis, return reduction strategy, or retention intervention.

    ---

    **4. Loyalty Strength**

    `loyalty_strength` represents the depth of the customer relationship with the brand.

    It combines:

    - `loyalty_program_member`
    - `premium_subscription`
    - `referral_count`
    - `brand_loyalty_score`

    These variables are selected because loyalty is not only about repeat purchases. Loyalty program membership and premium subscription show formal relationship signals. Referral count shows advocacy behavior, meaning the customer contributes beyond their own purchases. Brand loyalty score reflects the customer’s overall attachment to the brand.

    This score helps separate customers who are merely active from customers who have a stronger relationship with the business. Customers with high loyalty strength are usually better candidates for retention, advocacy, upsell, and long-term relationship strategies.

    ---

    **Why these four scores are used**

    The four scores are selected because they represent the main business dimensions needed for customer analytics:

    - `value_core` answers: **How valuable is the customer?**
    - `behavior_core` answers: **How active and engaged is the customer?**
    - `leakage_pressure` answers: **How much revenue friction or risk is present?**
    - `loyalty_strength` answers: **How strong is the relationship with the brand?**

    Together, these scores provide a more balanced view of customer quality. A customer can be high value but risky, highly engaged but low value, loyal but not very active, or promotion-sensitive with high leakage. By combining these dimensions, the analysis becomes more useful for segmentation, retention risk classification, revenue quality assessment, and business action prioritization.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **FM Segmentation**

    ---
    """)
    return


@app.cell
def _(df_segment, pd):
    # Create frequency and monetary value segments using quantile-based ranking to handle duplicate values safely
    df_segment["f_score"] = pd.qcut(
        df_segment["weekly_purchases"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Create monetary value segments using quantile-based ranking to handle duplicate values safely
    df_segment["m_score"] = pd.qcut(
        df_segment["monthly_spend"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Combine frequency and monetary scores into a single string for easier segment labeling
    df_segment["fm_score"] = (
        df_segment["f_score"].astype(str)
        + df_segment["m_score"].astype(str)
    )
    return


@app.cell
def _(pd):
    # Define a function to assign FM segments based on frequency and monetary scores
    def assign_fm_segment(row: pd.Series) -> str:
        f = row["f_score"]
        m = row["m_score"]

        if f >= 4 and m >= 4:
            return "High Frequency High Spend"
        if f >= 4 and m <= 2:
            return "Frequent Low Basket"
        if f <= 2 and m >= 4:
            return "High Spend Low Frequency"
        if f <= 2 and m <= 2:
            return "Low Frequency Low Spend"
        return "Mid Value Regulars"

    return (assign_fm_segment,)


@app.cell
def _(assign_fm_segment, df_segment):
    # Apply the function to create a new column for FM segments
    df_segment["fm_segment"] = df_segment.apply(assign_fm_segment, axis=1)

    # Display the distribution of FM segments to verify correct assignment
    fm_summary = (
        df_segment
        .groupby("fm_segment", dropna=False)
        .agg(
            customers=("user_id", "count"),
            avg_monthly_spend=("monthly_spend", "mean"),
            avg_weekly_purchases=("weekly_purchases", "mean"),
            avg_ltv=("lifetime_value_proxy", "mean"),
            avg_risk_adjusted_value=("risk_adjusted_value", "mean"),
        )
        .sort_values("customers", ascending=False)
    )

    # Calculate the percentage share of customers in each FM segment
    fm_summary["customer_share_pct"] = 100 * fm_summary["customers"] / len(df_segment)

    # Display the summary table with rounded values
    fm_summary.round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This step creates an FM segmentation framework to classify customers based on two core commercial dimensions:

    - `f_score`: frequency score from `weekly_purchases`
    - `m_score`: monetary score from `monthly_spend`
    - `fm_score`: combined Frequency and Monetary score
    - `fm_segment`: business-friendly customer segment label

    FM segmentation is adapted from the broader RFM framework, which is commonly used in customer analytics. RFM stands for Recency, Frequency, and Monetary. In this notebook, the segmentation focuses on Frequency and Monetary because the main objective is to understand how often customers buy and how much they spend.

    ---

    **1. Frequency Score**

    `f_score` measures how often a customer purchases.

    It is calculated from:

    - `weekly_purchases`

    Customers are ranked based on their weekly purchase frequency and then divided into five equal groups using quantile-based scoring. Customers in the lowest frequency group receive a score of 1, while customers in the highest frequency group receive a score of 5.

    This score helps identify customers who buy more often and show stronger purchase activity.

    ---

    **2. Monetary Score**

    `m_score` measures how much value a customer contributes.

    It is calculated from:

    - `monthly_spend`

    Customers are ranked based on their monthly spending and divided into five equal groups. Customers in the lowest spending group receive a score of 1, while customers in the highest spending group receive a score of 5.

    This score helps identify customers who contribute higher revenue to the business.

    ---

    **3. FM Score**

    `fm_score` combines the frequency score and monetary score into one label.

    For example:

    - `55` means the customer has both high purchase frequency and high spending
    - `51` means the customer buys frequently but spends relatively low
    - `15` means the customer buys less often but spends relatively high
    - `11` means the customer has both low purchase frequency and low spending

    This combined score gives a more complete view than looking at frequency or spending separately. A customer may purchase often but spend little per month, while another customer may purchase less frequently but spend more overall.

    ---

    **4. FM Segment**

    `fm_segment` translates the numeric FM score into a business-friendly label.

    This is important because business users usually need clear segment names rather than raw score combinations. The segment label makes the output easier to use in dashboards, reporting, and action planning.

    For example, customers with high frequency and high monetary value can be labeled as strong or valuable customers. Customers with low frequency and low monetary value can be treated as low-priority or low-engagement customers. Customers with mixed scores can be interpreted as potential growth, frequent low-value, or high-value low-frequency customers.

    ---

    **Why FM segmentation is used**

    FM segmentation is used because it provides a simple and practical way to understand customer value behavior.

    It answers two important business questions:

    - **Who buys more often?**
    - **Who spends more?**

    These two dimensions are directly related to revenue performance and customer prioritization. By using FM segmentation, the notebook can separate customers into groups that are easier to compare, monitor, and target with different business strategies.

    This segmentation also supports later steps in the analysis, especially behavioral segmentation, retention risk analysis, and business priority classification. For example, a customer with high monetary value but weak frequency may need a different strategy from a customer with high frequency but low spending.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Behavior Segmentation**

    ---
    """)
    return


@app.cell
def _(df_segment, qlabel):
    # Create quantile-based bands for the composite scores to facilitate behavior segment assignment
    df_segment["value_band"] = qlabel(df_segment["value_core"])
    df_segment["behavior_band"] = qlabel(df_segment["behavior_core"])
    df_segment["leakage_band"] = qlabel(df_segment["leakage_pressure"])
    df_segment["loyalty_band"] = qlabel(df_segment["loyalty_strength"])

    # Determine the threshold for high coupon dependency to identify promo-driven customers
    coupon_high_threshold = df_segment["coupon_dependency"].quantile(0.67)
    return (coupon_high_threshold,)


@app.cell
def _(coupon_high_threshold, df_segment, pd):
    # Define a function to assign behavior segments based on composite score bands and coupon dependency
    def assign_behavior_segment(row: pd.Series) -> str:
        if (
            row["value_band"] == "High"
            and row["leakage_band"] == "Low"
            and row["loyalty_band"] in ["Medium", "High"]
        ):
            return "Core Value Customers"

        if (
            row["behavior_band"] == "High"
            and row["loyalty_band"] == "High"
            and row["value_band"] in ["Medium", "High"]
        ):
            return "Loyal Frequent Buyers"

        if row["value_band"] == "High" and row["leakage_band"] == "High":
            return "High Value but Unstable"

        if (
            row["coupon_dependency"] >= coupon_high_threshold
            and row["value_band"] in ["Low", "Medium"]
        ):
            return "Promo-Driven Customers"

        if (
            row["behavior_band"] == "High"
            and row["value_band"] == "Medium"
            and row["leakage_band"] in ["Low", "Medium"]
        ):
            return "High Potential Customers"

        if row["value_band"] == "Low" and row["behavior_band"] == "Low":
            return "Low Value Low Engagement"

        return "General Customers"

    # Apply the function to create a new column for behavior segments
    df_segment["behavior_segment"] = df_segment.apply(assign_behavior_segment, axis=1)
    df_segment["behavior_segment"]
    return


@app.cell
def _(df_segment):
    # Display the distribution of behavior segments to verify correct assignment
    behavior_summary = (
        df_segment
        .groupby("behavior_segment", dropna=False)
        .agg(
            customers=("user_id", "count"),
            avg_monthly_spend=("monthly_spend", "mean"),
            avg_adjusted_revenue=("adjusted_revenue", "mean"),
            avg_risk_adjusted_value=("risk_adjusted_value", "mean"),
            avg_engagement=("engagement_score", "mean"),
            avg_leakage_pressure=("leakage_pressure", "mean"),
        )
        .sort_values("customers", ascending=False)
    )

    # Calculate the percentage share of customers in each behavior segment
    behavior_summary["customer_share_pct"] = 100 * behavior_summary["customers"] / len(df_segment)
    behavior_summary.round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This step creates customer segmentation based on five business dimensions:

    - `value_core`
    - `behavior_core`
    - `leakage_pressure`
    - `loyalty_strength`
    - `coupon_dependency`

    The purpose of this segmentation is to move beyond simple value-based grouping. Customers should not only be classified by how much they spend, but also by how they behave, how engaged they are, how much revenue leakage they create, how loyal they are, and how dependent they are on promotions.

    This makes the segmentation more useful for business decisions because two customers with similar spending levels may require very different strategies. For example, one customer may be high-value and loyal, while another may be high-value but unstable because of high returns, high abandonment, or strong coupon dependency.

    ---

    **1. Value Core**

    `value_core` represents the customer’s commercial contribution.

    This dimension helps identify whether a customer has strong revenue potential based on spending, lifetime value, adjusted revenue, and risk-adjusted value. Customers with high `value_core` are commercially important because they contribute more value to the business.

    This score is used because value is the foundation of customer prioritization. High-value customers may deserve stronger retention efforts, personalized offers, loyalty treatment, or proactive monitoring.

    ---

    **2. Behavior Core**

    `behavior_core` represents customer activity and engagement quality.

    This dimension captures how actively customers interact with the platform through session time, product views, app usage, wishlist behavior, notification response, engagement score, and browse-to-buy behavior.

    This score is used because engagement often indicates future business potential. A customer may not yet be among the highest spenders, but strong behavioral activity can suggest interest, intent, and growth opportunity.

    ---

    **3. Leakage Pressure**

    `leakage_pressure` represents revenue friction or value loss.

    This dimension captures behaviors such as cart abandonment, checkout abandonment, return rate, and return frequency. These signals show whether customer activity is translating into healthy revenue or being weakened by purchase friction and post-purchase returns.

    This score is used because high spending alone does not always mean strong revenue quality. A customer can appear valuable but still create business problems if they frequently abandon purchases or return products.

    ---

    **4. Loyalty Strength**

    `loyalty_strength` represents the depth of the customer relationship with the brand.

    This dimension combines loyalty program membership, premium subscription, referral behavior, and brand loyalty score. It helps separate customers who are only transactional from customers who show stronger attachment to the business.

    This score is used because loyal customers are often more suitable for retention, advocacy, premium treatment, and long-term relationship strategies.

    ---

    **5. Coupon Dependency**

    `coupon_dependency` represents how dependent customers are on discounts or promotions.

    This dimension is included separately because promotion dependency can affect margin quality and customer sustainability. A customer may generate revenue, but if they only purchase when coupons are available, their value may be less profitable or less stable.

    This signal is used because it helps identify promo-driven customers who may require discount control, margin protection, or alternative engagement strategies.

    ---

    **Why these dimensions are combined**

    These five dimensions are combined because they give a more complete customer view:

    - `value_core` answers: **How valuable is the customer?**
    - `behavior_core` answers: **How active and engaged is the customer?**
    - `leakage_pressure` answers: **How much revenue friction does the customer create?**
    - `loyalty_strength` answers: **How strong is the customer relationship?**
    - `coupon_dependency` answers: **How dependent is the customer on promotions?**

    Together, they allow the notebook to create more meaningful business segments, such as:

    - high-value and stable customers
    - loyal frequent buyers
    - high-potential customers
    - high-value but unstable customers
    - promo-driven customers
    - general customers
    - low-value low-engagement customers

    This segmentation is useful because each group requires a different business response. High-value stable customers should be protected, high-potential customers can be developed, promo-driven customers need margin control, and high-leakage or unstable customers may require retention or experience improvement actions.

    Overall, this step converts multiple customer behavior signals into business-friendly segment labels that can be used directly in Power BI dashboards, customer prioritization, and action planning.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Revenue Quality Label**

    ---
    """)
    return


@app.cell
def _(df_segment):
    # Analyze revenue quality across behavior segments by comparing adjusted revenue, return rates, and cart abandonment rates
    revenue_quality = (
        df_segment
        .groupby("behavior_segment", dropna=False)
        .agg(
            customers=("user_id", "count"),
            gross_revenue_proxy=("revenue_proxy", "mean"),
            adjusted_revenue=("adjusted_revenue", "mean"),
            revenue_efficiency=("revenue_efficiency", "mean"),
            avg_return_rate=("return_rate", "mean"),
            avg_cart_abandonment=("cart_abandonment_rate", "mean"),
            avg_coupon_dependency=("coupon_dependency", "mean"),
            avg_risk_adjusted_value=("risk_adjusted_value", "mean"),
        )
        .sort_values("adjusted_revenue", ascending=False)
    )
    return (revenue_quality,)


@app.cell
def _(np, revenue_quality):
    # Create a revenue quality label based on adjusted revenue, return rates, and cart abandonment rates
    revenue_quality["revenue_quality_label"] = np.where(
        (
            revenue_quality["adjusted_revenue"] >= revenue_quality["adjusted_revenue"].median()
        )
        & (
            revenue_quality["avg_return_rate"] <= revenue_quality["avg_return_rate"].median()
        )
        & (
            revenue_quality["avg_cart_abandonment"] <= revenue_quality["avg_cart_abandonment"].median()
        ),
        "Healthy Revenue",
        "Revenue with Leakage Pressure",
    )

    # Merge the revenue quality labels back to the main dataframe for further analysis and dashboard preparation
    revenue_quality_lookup = revenue_quality[["revenue_quality_label"]].reset_index()
    return (revenue_quality_lookup,)


@app.cell
def _(df_segment, revenue_quality_lookup):
    # Merge the revenue quality labels back to the main dataframe for further analysis and dashboard preparation
    df_segment_new = df_segment.merge(
        revenue_quality_lookup,
        on="behavior_segment",
        how="left",
    )

    # Display the behavior segments along with their assigned revenue quality labels to verify correct merging
    df_segment_new[["behavior_segment", "revenue_quality_label"]].head()
    return (df_segment_new,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This step creates a `revenue_quality_label` to classify whether each customer segment generates healthy revenue or revenue with leakage pressure.

    The label is first created at the `behavior_segment` level, not directly at the individual customer level. This is because revenue quality is easier to interpret when customers are grouped by similar behavioral characteristics. Instead of judging each customer in isolation, this approach evaluates whether a segment as a whole produces strong and efficient revenue with manageable leakage signals.

    ---

    **Revenue quality aggregation**

    The data is grouped by `behavior_segment`, then several business metrics are calculated:

    - `customers`: number of customers in each segment
    - `gross_revenue_proxy`: average gross revenue proxy
    - `adjusted_revenue`: average adjusted revenue
    - `revenue_efficiency`: average revenue efficiency
    - `avg_return_rate`: average return rate
    - `avg_cart_abandonment`: average cart abandonment rate
    - `avg_coupon_dependency`: average coupon dependency
    - `avg_risk_adjusted_value`: average risk-adjusted value

    These metrics are used to evaluate whether revenue from each segment is strong, efficient, and relatively clean from leakage.

    ---

    **Why adjusted revenue is used**

    `adjusted_revenue` is used as the main revenue quality metric because it provides a cleaner view of customer revenue than gross revenue alone.

    Gross revenue can look strong even when customers create operational or commercial problems, such as frequent returns, high abandonment, or high promotion dependency. Adjusted revenue helps provide a more realistic view of revenue contribution after considering business quality factors.

    ---

    **Why return rate and cart abandonment are included**

    Revenue quality is not only about how much revenue a segment generates. It also depends on how much revenue friction exists.

    Two leakage indicators are used in the label logic:

    - `avg_return_rate`
    - `avg_cart_abandonment`

    `avg_return_rate` represents post-purchase leakage. A high return rate means some completed purchases may not fully become retained revenue.

    `avg_cart_abandonment` represents pre-purchase leakage. A high cart abandonment rate means customers show purchase intent but do not complete the transaction.

    By combining revenue value and leakage indicators, the notebook can separate segments that generate healthy revenue from segments that may appear valuable but have hidden revenue problems.

    ---

    **Labeling logic**

    A segment is labeled as `Healthy Revenue` when it meets all of these conditions:

    - its `adjusted_revenue` is greater than or equal to the median adjusted revenue
    - its `avg_return_rate` is less than or equal to the median return rate
    - its `avg_cart_abandonment` is less than or equal to the median cart abandonment rate

    If a segment does not meet all of these conditions, it is labeled as:

    `Revenue with Leakage Pressure`

    This means the segment may still generate revenue, but that revenue is exposed to issues such as returns, abandonment, or lower revenue quality.

    ---

    **Why the label is created at segment level**

    The `revenue_quality_label` is created at the `behavior_segment` level because the goal is to describe the revenue quality profile of each segment.

    This makes the result easier to interpret in a Power BI dashboard. Business users can compare segments and quickly identify which groups produce healthier revenue and which groups need attention because of leakage pressure.

    After the label is created, it is merged back into the customer-level table. This allows every customer to inherit the revenue quality label of their behavior segment.

    ---

    **Why it is merged back to customer level**

    Merging the label back into `df_segment` is important because the final Power BI fact table is customer-level.

    By adding `revenue_quality_label` to each customer row, the dashboard can support both:

    - segment-level analysis
    - customer-level filtering and slicing

    This means users can analyze revenue quality by behavior segment, country, income level, risk flag, business priority, or any other customer attribute in the final dataset.

    ---

    **Business interpretation**

    This step helps answer an important business question:

    **Which customer segments generate strong and clean revenue, and which segments generate revenue with leakage pressure?**

    A `Healthy Revenue` segment is generally more desirable because it combines stronger adjusted revenue with lower return and abandonment pressure.

    A `Revenue with Leakage Pressure` segment may still be commercially important, but it requires closer attention. Possible business responses include improving checkout experience, reducing return behavior, reviewing product fit, controlling discount dependency, or designing more targeted retention actions.

    Overall, this step improves the dashboard by adding a business-friendly revenue quality classification that connects customer behavior, revenue contribution, and leakage risk.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Retention Risk Score dan Risk Flag**

    ---
    """)
    return


@app.cell
def _(MinMaxScaler, df_segment_new):
    # Create a helper function for single-column Min-Max scaling
    def minmax_scaler_series(series):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(series.to_frame()).flatten()

    # Retention risk score calculation
    df_segment_new["risk_score_norm"] = minmax_scaler_series(df_segment_new["risk_score"])
    df_segment_new["abandonment_norm"] = minmax_scaler_series(df_segment_new["cart_abandonment_rate"])
    df_segment_new["checkout_abandonment_norm"] = minmax_scaler_series(df_segment_new["checkout_abandonments_per_month"])
    df_segment_new["return_rate_norm"] = minmax_scaler_series(df_segment_new["return_rate"])
    df_segment_new["coupon_dependency_norm"] = minmax_scaler_series(df_segment_new["coupon_dependency"])
    df_segment_new["churn_proxy_norm"] = minmax_scaler_series(df_segment_new["churn_proxy"])
    df_segment_new["stability_inverse"] = 1 - minmax_scaler_series(df_segment_new["stability_score"])
    df_segment_new["retention_strength_inverse"] = 1 - minmax_scaler_series(df_segment_new["retention_strength"])

    df_segment_new["retention_risk_score"] = (
        0.22 * df_segment_new["risk_score_norm"] +
        0.16 * df_segment_new["abandonment_norm"] +
        0.12 * df_segment_new["checkout_abandonment_norm"] +
        0.14 * df_segment_new["return_rate_norm"] +
        0.12 * df_segment_new["coupon_dependency_norm"] +
        0.10 * df_segment_new["churn_proxy_norm"] +
        0.07 * df_segment_new["stability_inverse"] +
        0.07 * df_segment_new["retention_strength_inverse"]
    )

    # Determine quantile thresholds for retention risk score to create risk segments
    q1_1 = df_segment_new["retention_risk_score"].quantile(0.50)
    q2_1 = df_segment_new["retention_risk_score"].quantile(0.80)
    q3_1 = df_segment_new["retention_risk_score"].quantile(0.95)
    return q1_1, q2_1, q3_1


@app.cell
def _(df_segment_new, q1_1, q2_1, q3_1):
    # Define function to assign risk flags based on quantiles
    def assign_risk_flag(score):
        if score >= q3_1:
            return "Critical Risk"
        elif score >= q2_1:
            return "At Risk"
        elif score >= q1_1:
            return "Watchlist"
        else:
            return "Stable"

    df_segment_new["retention_risk_flag"] = df_segment_new["retention_risk_score"].apply(assign_risk_flag)
    df_segment_new[["retention_risk_score", "retention_risk_flag"]].head()
    return


@app.cell
def _(df_segment_new):
    # Analyze retention risk distribution
    risk_distribution = (
        df_segment_new.groupby("retention_risk_flag")
          .agg(
              customers=("user_id", "count"),
              avg_monthly_spend=("monthly_spend", "mean"),
              avg_risk_adjusted_value=("risk_adjusted_value", "mean"),
              avg_return_rate=("return_rate", "mean"),
              avg_cart_abandonment=("cart_abandonment_rate", "mean"),
              avg_coupon_dependency=("coupon_dependency", "mean")
          )
          .sort_index()
    )

    # Calculate customer share percentage for each risk category
    risk_distribution["customer_share_pct"] = 100 * risk_distribution["customers"] / len(df_segment_new)
    risk_distribution.round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This step creates a `retention_risk_score` to estimate how exposed each customer is to potential retention problems.

    The score is built as a weighted composite metric using several risk-related indicators. Each variable is first normalized using Min-Max scaling so that all inputs are converted into the same 0-1 range. This is important because the original variables have different units, scales, and distributions.

    After normalization, the variables are combined using business-defined weights. The result is a single retention risk score where a higher value indicates a higher level of customer risk.

    ---

    **Variables used in the retention risk score**

    The score is calculated from the following components:

    - `risk_score`
    - `cart_abandonment_rate`
    - `checkout_abandonments_per_month`
    - `return_rate`
    - `coupon_dependency`
    - `churn_proxy`
    - inverse of `stability_score`
    - inverse of `retention_strength`

    These variables are selected because they represent different types of retention risk: general customer risk, purchase friction, post-purchase leakage, promotion dependency, churn signal, weak stability, and weak retention strength.

    ---

    **1. Risk Score**

    `risk_score` receives the largest weight because it is the main risk indicator already available in the dataset.

    It summarizes the customer’s general risk profile and provides a broad signal of potential instability. Since it is the most direct risk-related variable, it has the highest influence in the final retention risk score.

    ---

    **2. Cart Abandonment**

    `cart_abandonment_rate` is included because it reflects purchase intent that does not convert into completed transactions.

    A customer with high cart abandonment may still be interested in the product, but there may be friction in pricing, product fit, checkout experience, shipping cost, or decision confidence. This behavior can indicate weakening purchase commitment.

    ---

    **3. Checkout Abandonment**

    `checkout_abandonments_per_month` is also included because checkout abandonment happens closer to the final transaction stage.

    This signal is important because the customer has already moved beyond browsing and cart activity. If customers frequently abandon at checkout, the business may be losing revenue at a critical stage of the funnel.

    ---

    **4. Return Behavior**

    `return_rate` is included as a post-purchase risk signal.

    High return behavior can reduce revenue quality and may indicate dissatisfaction, poor product fit, wrong expectations, or unstable purchasing behavior. Even if a customer buys often, frequent returns can weaken their long-term value.

    ---

    **5. Coupon Dependency**

    `coupon_dependency` is included because customers who rely heavily on discounts may be less stable and less profitable.

    A highly coupon-dependent customer may continue purchasing only when incentives are available. This creates margin pressure and can make retention weaker if discounts are reduced.

    ---

    **6. Churn Proxy**

    `churn_proxy` is included as an early warning indicator of potential customer decline.

    This variable helps capture customers who may already show signals associated with churn risk. Including it makes the score more forward-looking rather than only describing current behavior.

    ---

    **7. Inverse Stability**

    `stability_score` is inverted using:

    `1 - normalized stability_score`

    This means lower stability becomes higher risk.

    This transformation is needed because `stability_score` is a positive health indicator. A high stability score means the customer is more stable, while a low stability score means the customer is more uncertain or inconsistent. By using the inverse, the direction becomes aligned with the risk score: higher value means higher risk.

    ---

    **8. Inverse Retention Strength**

    `retention_strength` is also inverted using:

    `1 - normalized retention_strength`

    This means weaker retention strength becomes higher risk.

    This transformation is used for the same reason as stability. Retention strength is a positive customer health metric, so it must be reversed before being included in a risk score.

    ---

    **Weighting rationale**

    The weights are assigned based on business importance:

    - `risk_score`: 0.22
    - `cart_abandonment_rate`: 0.16
    - `checkout_abandonments_per_month`: 0.12
    - `return_rate`: 0.14
    - `coupon_dependency`: 0.12
    - `churn_proxy`: 0.10
    - inverse `stability_score`: 0.07
    - inverse `retention_strength`: 0.07

    The largest weight is assigned to `risk_score` because it is the main risk indicator. Abandonment and return behavior also receive meaningful weights because they directly represent revenue friction and customer dissatisfaction signals. Coupon dependency and churn proxy are included as supporting risk indicators. Stability and retention strength receive smaller weights because they are health indicators that complement the main risk variables.

    The total weight equals 1, which makes the final score easier to interpret as a normalized weighted composite risk score.

    ---

    **Risk flag classification**

    After calculating `retention_risk_score`, customers are classified into four risk groups using quantile thresholds:

    - `Stable`
    - `Watchlist`
    - `At Risk`
    - `Critical Risk`

    The thresholds are based on the distribution of the risk score:

    - below the 50th percentile → `Stable`
    - 50th to 80th percentile → `Watchlist`
    - 80th to 95th percentile → `At Risk`
    - 95th percentile and above → `Critical Risk`

    This approach creates relative risk groups based on the customer population. It is useful when there is no fixed business threshold available yet.

    ---

    **Business interpretation**

    The `retention_risk_score` helps identify which customers require closer attention.

    A `Stable` customer has relatively low retention risk compared with the rest of the customer base.

    A `Watchlist` customer is not yet critical, but already shows moderate risk signals and should be monitored.

    An `At Risk` customer shows stronger warning signs and may need targeted retention actions.

    A `Critical Risk` customer belongs to the highest-risk group and should be prioritized for urgent intervention.

    ---

    **Why this score is useful**

    This score is useful because retention risk is usually not caused by one single variable. A customer may be risky because they abandon carts, return products, depend on coupons, show churn signals, or have weak stability.

    By combining these signals into one composite score, the notebook creates a more complete and practical view of customer risk. This score can be used in Power BI to monitor risk distribution, compare risky and stable customers, prioritize business actions, and support retention strategy.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Business Priority**

    ---
    """)
    return


@app.cell
def _(df_segment_new, pd):
    # Define a function to assign business priorities based on behavior segments and retention risk flags
    def assign_business_priority(row: pd.Series) -> str:
        seg = row["behavior_segment"]
        risk = row["retention_risk_flag"]

        if seg in ["Core Value Customers", "Loyal Frequent Buyers"] and risk == "Stable":
            return "Protect and Retain"

        if seg == "High Value but Unstable" or risk == "Critical Risk":
            return "Urgent Retention Intervention"

        if seg == "High Potential Customers" and risk in ["Stable", "Watchlist"]:
            return "Upsell and Value Expansion"

        if seg == "Promo-Driven Customers":
            return "Discount Control and Margin Protection"

        if seg == "Low Value Low Engagement":
            return "Low-Cost Automation"

        if risk == "At Risk":
            return "Retention Watchlist"

        return "Maintain and Monitor"


    # Apply the function to create a new column for business priorities
    df_segment_new["business_priority"] = df_segment_new.apply(assign_business_priority, axis=1)
    df_segment_new["business_priority"]
    return


@app.cell
def _(df_segment, df_segment_new):
    # Display the distribution of business priorities to verify correct assignment
    priority_summary = (
        df_segment_new
        .groupby("business_priority", dropna=False)
        .agg(
            customers=("user_id", "count"),
            avg_monthly_spend=("monthly_spend", "mean"),
            avg_engagement=("engagement_score", "mean"),
            avg_retention_risk=("retention_risk_score", "mean"),
            avg_adjusted_revenue=("adjusted_revenue", "mean"),
        )
        .sort_values("customers", ascending=False)
    )

    # Calculate the percentage share of customers in each business priority category
    priority_summary["customer_share_pct"] = 100 * priority_summary["customers"] / len(df_segment)
    priority_summary["customer_share_pct"]
    return (priority_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This step combines `behavior_segment` and `retention_risk_flag` into a business action priority label.

    The purpose is to translate analytical segmentation into practical business decisions. A customer segment alone explains customer behavior, while a risk flag explains retention exposure. When both are combined, the output becomes more actionable because it shows not only **who the customer is**, but also **what kind of action the business should take**.

    ---

    **Why behavior segment and retention risk are combined**

    `behavior_segment` describes the customer’s business profile, such as whether the customer is high-value, loyal, promo-driven, high-potential, or low-engagement.

    `retention_risk_flag` describes the customer’s risk condition, such as whether the customer is stable, needs monitoring, is at risk, or is critically risky.

    Combining these two variables helps create a more complete prioritization framework.

    For example:

    - A high-value stable customer should be protected.
    - A high-value unstable customer needs urgent intervention.
    - A high-potential customer should be developed through upsell or value expansion.
    - A promo-driven customer needs discount and margin control.
    - A low-value low-engagement customer should not receive expensive manual treatment.

    This makes the output more useful for CRM, retention planning, marketing strategy, and Power BI dashboard reporting.

    ---

    **Business priority logic**

    The logic assigns each customer into one of several action groups:

    **1. Protect and Retain**

    Customers are assigned to `Protect and Retain` when they belong to:

    - `Core Value Customers`
    - `Loyal Frequent Buyers`

    and their risk flag is:

    - `Stable`

    These customers are valuable and healthy. The business should protect them through loyalty treatment, personalized engagement, premium service, or relationship-building campaigns.

    ---

    **2. Urgent Retention Intervention**

    Customers are assigned to `Urgent Retention Intervention` when they are:

    - `High Value but Unstable`

    or have risk flag:

    - `Critical Risk`

    These customers require immediate attention because they either generate strong value with instability or show the highest level of retention risk. The business should prioritize them for targeted retention actions, service recovery, personalized offers, or customer success intervention.

    ---

    **3. Upsell and Value Expansion**

    Customers are assigned to `Upsell and Value Expansion` when they are:

    - `High Potential Customers`

    and their risk flag is:

    - `Stable`
    - `Watchlist`

    These customers are not necessarily the highest-value customers yet, but they show potential. The recommended strategy is to increase their value through cross-sell, upsell, product recommendations, loyalty activation, or personalized campaigns.

    ---

    **4. Discount Control and Margin Protection**

    Customers are assigned to `Discount Control and Margin Protection` when they are:

    - `Promo-Driven Customers`

    These customers may rely heavily on coupons or promotions. The business should avoid over-discounting and focus on margin protection, alternative incentives, bundle strategies, or non-discount engagement.

    ---

    **5. Low-Cost Automation**

    Customers are assigned to `Low-Cost Automation` when they are:

    - `Low Value Low Engagement`

    These customers have low commercial value and weak engagement. They should not be prioritized for costly manual retention actions. A more efficient approach is automated email, low-cost campaigns, lifecycle nudges, or passive reactivation flows.

    ---

    **6. Retention Watchlist**

    Customers are assigned to `Retention Watchlist` when their risk flag is:

    - `At Risk`

    This group shows meaningful retention risk but may not always fall into the most urgent segment logic. They should be monitored and targeted with preventive retention actions before they move into critical risk.

    ---

    **7. Maintain and Monitor**

    Customers who do not meet the specific priority rules are assigned to:

    - `Maintain and Monitor`

    This is the default group. These customers do not require urgent action, but they should still be monitored through regular dashboard tracking and standard CRM campaigns.

    ---

    **Why this step is important**

    This step turns segmentation into an action framework.

    Without `business_priority`, the dashboard would only describe customer groups. With `business_priority`, the dashboard can help answer a more practical question:

    **What should the business do with each customer group?**

    This makes the analysis more useful for decision-making because each customer receives a clear action category that can be used for:

    - executive overview
    - customer prioritization
    - retention planning
    - campaign targeting
    - CRM action playbooks
    - Power BI filtering and reporting

    ---

    **Output interpretation**

    The `priority_summary` table validates the assigned priorities by showing:

    - number of customers in each priority group
    - average monthly spend
    - average engagement
    - average retention risk
    - average adjusted revenue
    - customer share percentage

    This summary helps evaluate whether the priority groups make business sense. For example, urgent intervention groups should generally show higher risk, while protect-and-retain groups should show healthier customer profiles. Low-cost automation groups should usually have weaker value or engagement.

    Overall, this step bridges the gap between analytics and business action by converting customer behavior and risk signals into clear operational priorities.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Segment Action Playbook**

    ---
    """)
    return


@app.cell
def _(df_segment_new, pd):
    # Display the summary table with rounded values
    playbook = pd.DataFrame({
        "behavior_segment": [
            "Core Value Customers",
            "Loyal Frequent Buyers",
            "High Value but Unstable",
            "Promo-Driven Customers",
            "High Potential Customers",
            "Low Value Low Engagement",
            "General Customers",
        ],
        "main_characteristic": [
            "High value, low leakage, relatively stable",
            "High activity, loyal relationship, repeat-oriented",
            "Strong value but elevated leakage and retention pressure",
            "Dependent on coupons, weaker standalone value quality",
            "Good engagement and room to grow value",
            "Weak value and weak engagement",
            "Average profile without dominant strength or risk",
        ],
        "business_objective": [
            "Protect loyalty and preserve revenue quality",
            "Maintain habit and increase wallet share",
            "Reduce leakage and prevent value decline",
            "Control discount dependence and protect margin",
            "Increase monetization through targeted growth",
            "Maintain efficiently at low cost",
            "Monitor and optimize selectively",
        ],
        "recommended_action": [
            "VIP retention, premium service, personalized appreciation",
            "Cross-sell, membership benefits, frequency reinforcement",
            "Proactive outreach, service recovery, return control, checkout recovery",
            "Reduce blanket discounts, test non-price incentives, tighten promo targeting",
            "Personalized upsell, category expansion, bundle recommendation",
            "Automation-only journeys, low-cost communication, passive nurture",
            "Routine monitoring and light segmentation treatment",
        ],
    })

    # Merge the playbook recommendations back to the main dataframe for further analysis and dashboard preparation
    df_segment_new_2 = df_segment_new.merge(
        playbook[["behavior_segment", "recommended_action"]],
        on="behavior_segment",
        how="left",
    )

    df_segment_new_2[["behavior_segment", "recommended_action"]].head()
    return df_segment_new_2, playbook


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This step creates a segment-level action playbook and merges the recommended action back into the customer-level fact table.

    The playbook is created at the `behavior_segment` level because each behavior segment represents a distinct customer profile. Instead of assigning actions manually to every individual customer, the notebook defines a clear business treatment for each segment group. This makes the recommendation logic easier to maintain, explain, and use in Power BI.

    ---

    **Purpose of the playbook**

    The playbook translates customer segmentation into practical business actions.

    Each `behavior_segment` is mapped to three business interpretation fields:

    - `main_characteristic`
    - `business_objective`
    - `recommended_action`

    These fields help explain what each segment means, what the business should focus on, and what type of action is most appropriate.

    This is important because segmentation should not stop at labeling customers. A useful segmentation framework should also guide decision-making.

    ---

    **Segment-level recommendation logic**

    Each behavior segment receives a different recommended action based on its commercial value, engagement quality, leakage pressure, loyalty strength, and growth potential.

    For example:

    - `Core Value Customers` receive VIP retention and personalized appreciation because they are high-value and relatively stable.
    - `Loyal Frequent Buyers` receive cross-sell, membership benefits, and frequency reinforcement because they already show strong habits and loyalty.
    - `High Value but Unstable` customers receive proactive outreach, service recovery, return control, and checkout recovery because they generate value but also show leakage or retention pressure.
    - `Promo-Driven Customers` receive discount control and non-price incentive strategies because their value may depend too heavily on promotions.
    - `High Potential Customers` receive upsell, category expansion, and bundle recommendations because they show room for value growth.
    - `Low Value Low Engagement` customers receive low-cost automated treatment because expensive manual retention actions may not be efficient.
    - `General Customers` receive routine monitoring and light segmentation treatment because they do not show a dominant strength or risk pattern.

    ---

    **Why the playbook is created at segment level**

    The playbook is created at the segment level because the goal is to provide consistent business treatment for customers with similar behavior profiles.

    This approach has several advantages:

    - easier to explain to stakeholders
    - easier to maintain than customer-by-customer manual rules
    - more suitable for dashboard storytelling
    - more consistent for CRM and campaign planning
    - easier to update when business strategy changes

    If the strategy changes later, the recommended action can be updated in the playbook table without changing the entire segmentation logic.

    ---

    **Why recommended action is merged back to the fact table**

    After the playbook is created, `recommended_action` is merged back into the customer-level table.

    This is necessary because the final Power BI fact table is built at customer level. By adding `recommended_action` to every customer row, the dashboard can support more flexible analysis, such as:

    - customer count by recommended action
    - average monthly spend by recommended action
    - average retention risk by recommended action
    - adjusted revenue by recommended action
    - recommended action by country, income level, gender, or risk flag
    - business priority and action playbook views

    This allows the dashboard to connect customer analytics directly with operational action.

    ---

    **Business interpretation**

    This step helps answer the question:

    **What should the business do with each customer segment?**

    The output provides an action-oriented layer on top of segmentation. Instead of only showing that a customer belongs to a specific behavior segment, the final table also shows what type of treatment is recommended.

    This makes the dashboard more useful for business users because it connects analytical findings with practical CRM, retention, marketing, and revenue-quality decisions.

    Overall, the playbook turns the segmentation output into a simple action framework that can be used directly in Power BI reporting and business prioritization.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Make Fact & Dimension Tables**

    ---
    """)
    return


@app.cell
def _(df_segment_new_2, require_columns):
    # Define the columns to be included in the fact table for the customer dashboard
    fact_columns = [
        # Grain and slicers
        "user_id",
        "country",
        "income_level",
        "gender",
        "urban_rural",
        "premium_subscription",
        "loyalty_program_member",

        # Optional profile labels from Shopper notebook
        "premium_subscription_label",
        "loyalty_program_member_label",
        "customer_group",

        # Value and revenue
        "monthly_spend",
        "weekly_purchases",
        "average_order_value",
        "lifetime_value_proxy",
        "risk_adjusted_value",
        "adjusted_revenue",
        "revenue_efficiency",

        # Engagement
        "engagement_score",
        "daily_session_time_minutes",
        "product_views_per_day",
        "app_usage_frequency",
        "browse_to_buy_ratio",
        "notification_response_rate",

        # Leakage and risk
        "return_rate",
        "return_frequency",
        "cart_abandonment_rate",
        "checkout_abandonments_per_month",
        "coupon_dependency",
        "coupon_usage_frequency",
        "risk_score",
        "risk_exposure",
        "stability_score",
        "retention_strength",
        "churn_proxy",
        "retention_risk_score",
        "retention_risk_flag",

        # Loyalty
        "brand_loyalty_score",
        "referral_count",

        # Segmentation
        "f_score",
        "m_score",
        "fm_score",
        "fm_segment",
        "behavior_segment",
        "revenue_quality_label",
        "business_priority",
        "recommended_action",
    ]

    # Validate that all required columns for the fact table are present in the dataframe
    require_columns(df_segment_new_2, fact_columns, step_name="fact_customer_dashboard")

    # Create the fact table for the customer dashboard by selecting the relevant columns from the main dataframe
    fact_customer_dashboard = df_segment_new_2[fact_columns].copy()

    # Display the shape and a sample of the fact table to verify correct preparation
    print("Fact shape:", fact_customer_dashboard.shape)
    fact_customer_dashboard.head()
    return (fact_customer_dashboard,)


@app.cell
def _(pd):
    # Create dimension tables for retention risk, business priority, and behavior segments to support dashboard filtering and labeling
    dim_retention_risk = pd.DataFrame({
        "retention_risk_flag": [
            "Stable",
            "Watchlist",
            "At Risk",
            "Critical Risk",
        ],
        "risk_order": [1, 2, 3, 4],
        "risk_color": [
            "#6CCB63",
            "#F4C542",
            "#F28E2B",
            "#D62728",
        ],
        "risk_definition": [
            "Low relative retention risk",
            "Moderate early warning risk",
            "High retention risk",
            "Severe retention risk requiring urgent intervention",
        ],
    })

    # Create a dimension table for business priorities with associated groups and action owners to facilitate dashboard organization and user guidance
    dim_business_priority = pd.DataFrame({
        "business_priority": [
            "Protect and Retain",
            "Urgent Retention Intervention",
            "Upsell and Value Expansion",
            "Discount Control and Margin Protection",
            "Retention Watchlist",
            "Low-Cost Automation",
            "Maintain and Monitor",
        ],
        "priority_order": [1, 2, 3, 4, 5, 6, 7],
        "priority_group": [
            "Retention Protection",
            "Risk Intervention",
            "Growth Expansion",
            "Margin Protection",
            "Risk Monitoring",
            "Low-Cost Management",
            "Business as Usual",
        ],
        "action_owner": [
            "CRM / Loyalty",
            "Retention / Customer Success",
            "Growth / CRM",
            "Marketing / Pricing",
            "Retention Team",
            "Marketing Automation",
            "CRM Analytics",
        ],
    })

    # Create a dimension table for behavior segments with detailed descriptions and color coding to enhance dashboard visualization and user understanding
    dim_behavior_segment = pd.DataFrame({
        "behavior_segment": [
            "Core Value Customers",
            "Loyal Frequent Buyers",
            "High Value but Unstable",
            "High Potential Customers",
            "Promo-Driven Customers",
            "General Customers",
            "Low Value Low Engagement",
        ],
        "segment_order": [1, 2, 3, 4, 5, 6, 7],
        "segment_group": [
            "High Value",
            "Loyal / Frequent",
            "High Value Risk",
            "Growth Potential",
            "Promo Sensitive",
            "General",
            "Low Value",
        ],
        "segment_description": [
            "High value, low leakage, relatively stable",
            "High activity and loyal repeat-oriented customers",
            "Strong value but elevated leakage and risk pressure",
            "Good engagement with room to grow value",
            "Customers dependent on coupons or promotions",
            "Average profile without dominant strength or risk",
            "Weak value and weak engagement",
        ],
        "segment_color": [
            "#1F77B4",
            "#2CA02C",
            "#D62728",
            "#9467BD",
            "#FF7F0E",
            "#7F7F7F",
            "#BCBD22",
        ],
    })
    return dim_behavior_segment, dim_business_priority, dim_retention_risk


@app.cell
def _(dim_retention_risk):
    # Display the retention risk dimension table to verify correct creation
    dim_retention_risk.head()
    return


@app.cell
def _(dim_business_priority):
    # Display the behavior segment dimension table to verify correct creation
    dim_business_priority.head()
    return


@app.cell
def _(dim_behavior_segment):
    # Display the behavior segment dimension table to verify correct creation
    dim_behavior_segment.head()
    return


@app.cell
def _(
    dim_behavior_segment,
    dim_business_priority,
    dim_retention_risk,
    fact_customer_dashboard,
    pd,
):
    # Define a function to validate that all dimension keys used in the fact table
    # are available in the corresponding dimension table.
    # This helps ensure data integrity and prevents relationship issues in Power BI.
    def validate_dimension_key(
        fact_df: pd.DataFrame,
        dim_df: pd.DataFrame,
        key: str,
        dim_name: str,
    ) -> None:
        fact_values = set(fact_df[key].dropna().unique())
        dim_values = set(dim_df[key].dropna().unique())
        missing_values = sorted(fact_values - dim_values)

        if missing_values:
            raise ValueError(
                f"The {dim_name} dimension table does not contain values for {key}: {missing_values}"
            )

        print(f"OK - {dim_name}: all values in {key} are covered.")


    # Validate that all retention risk flags in the fact table
    # are available in the retention risk dimension table.
    validate_dimension_key(
        fact_customer_dashboard,
        dim_retention_risk,
        "retention_risk_flag",
        "dim_retention_risk",
    )


    # Validate that all business priority values in the fact table
    # are available in the business priority dimension table.
    validate_dimension_key(
        fact_customer_dashboard,
        dim_business_priority,
        "business_priority",
        "dim_business_priority",
    )

    # Validate that all behavior segment values in the fact table
    # are available in the behavior segment dimension table.
    validate_dimension_key(
        fact_customer_dashboard,
        dim_behavior_segment,
        "behavior_segment",
        "dim_behavior_segment",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Fact and dim successfully created

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Export CSV**

    ---
    """)
    return


@app.cell
def _(
    dim_behavior_segment,
    dim_business_priority,
    dim_retention_risk,
    fact_customer_dashboard,
    playbook,
    priority_summary,
):
    # Export the fact and dimension tables to CSV files for use in Power BI, ensuring that dimension keys are consistent and data integrity is maintained for accurate dashboard relationships and filtering.
    export_tables = {
        "fact_customer_dashboard.csv": (fact_customer_dashboard, False),
        "dim_retention_risk.csv": (dim_retention_risk, False),
        "dim_business_priority.csv": (dim_business_priority, False),
        "dim_behavior_segment.csv": (dim_behavior_segment, False),
        "business_priority_summary.csv": (priority_summary, True),
        "segment_action_playbook.csv": (playbook, False),
    }

    for file_name, (table, include_index) in export_tables.items():
        table.to_csv(file_name, index=include_index)
    return


@app.cell
def _(fact_customer_dashboard):
    print("Fact table shape:", fact_customer_dashboard.shape)
    print("Fact table columns:")
    for col in fact_customer_dashboard.columns:
        print("-", col)

    fact_customer_dashboard.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## **Conclusion**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the following relationships in Power BI:

    | From table | From column | To table | To column | Relationship |
    |---|---|---|---|---|
    | `fact_customer_dashboard` | `retention_risk_flag` | `dim_retention_risk` | `retention_risk_flag` | Many-to-one |
    | `fact_customer_dashboard` | `business_priority` | `dim_business_priority` | `business_priority` | Many-to-one |
    | `fact_customer_dashboard` | `behavior_segment` | `dim_behavior_segment` | `behavior_segment` | Many-to-one |

    Set the filter direction to: **single direction from dimension table to fact table**.

    Sorting:
    - Sort `retention_risk_flag` by `risk_order`
    - Sort `business_priority` by `priority_order`
    - Sort `behavior_segment` by `segment_order`

    ---
    """)
    return


if __name__ == "__main__":
    app.run()
