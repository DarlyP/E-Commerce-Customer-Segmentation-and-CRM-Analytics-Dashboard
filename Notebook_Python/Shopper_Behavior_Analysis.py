import marimo

__generated_with = "0.23.4"
app = marimo.App(auto_download=["html", "ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **E-Commerce Customer Segmentation and CRM Analytics Dashboard - Shopper Behavior Analysis**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    >**The aim of this notebook:**
    > - Analyze customer behavior in ways that are relevant to business performance
    > - Identify patterns related to customer value, activity, responsiveness, and risk
    > - Explore differences across customer groups to support strategic decision-making
    > - Generate insights that can guide future segmentation and retention initiatives

    > **Expected Output:**
    > - A business-focused view of customer behavior across key dimensions
    > - Insights into spending patterns and transaction intensity
    > - Insights into marketing responsiveness and engagement behavior
    > - Identification of customer risk signals such as cart abandonment and return behavior
    > - Group-level comparisons for loyalty and premium customers
    > - Early business insights to support segmentation and customer strategy

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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set display options for better readability
    from IPython.display import display

    return display, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## **Data Loading**

    ---
    """)
    return


@app.cell
def _(pd):
    # Load the dataset
    df = pd.read_csv(r"C:\Users\user\Documents\Coding\Portofolio\E-Commerce-Customer-Segmentation-and-CRM-Analytics-Dashboard\Dataset\customer_data_selected.csv")
    df.head()
    return (df,)


@app.cell
def _(df):
    # Check the columns in the dataset
    print(df.columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Data successfully loading

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Data Preparation**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Business question:**
    Do we have a clean and business-ready dataset that supports value, activity, marketing, and risk analysis?

    ---
    """)
    return


@app.cell
def _(df):
    # Check the data types and missing values
    df.info()
    return


@app.cell
def _(df):
    # Define the required columns based on the dataset description
    required_cols = [
        'monthly_spend', 'weekly_purchases', 'average_order_value', 'age',
        'gender', 'country', 'income_level', 'income_group',
        'employment_status', 'education_level', 'household_size',
        'daily_session_time_minutes', 'product_views_per_day',
        'app_usage_frequency', 'wishlist_items_count', 'engagement_score',
        'ad_views_per_day', 'ad_clicks_per_day', 'notification_response_rate',
        'coupon_usage_frequency', 'ctr', 'coupon_dependency',
        'cart_abandonment_rate', 'checkout_abandonments_per_month',
        'return_rate', 'return_frequency', 'risk_score',
        'loyalty_program_member', 'premium_subscription', 'account_age_months',
        'last_purchase_date', 'purchase_month', 'purchase_day', 'day_of_week',
        'revenue_proxy', 'lifetime_value_proxy', 'risk_adjusted_value',
        'spend_per_purchase', 'spend_to_aov_ratio', 'spend_per_view',
        'engagement_spend', 'risk_spend'
    ]

    # Check for missing columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check for missing values in the dataset
    missing
    return


@app.cell
def _(df):
    # Calculate the number of missing values in each column
    df_missing_values = df.isnull().sum()
    df_missing_values
    return


@app.cell
def _(df, pd):
    # Convert 'last_purchase_date' to datetime format
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
    df['last_purchase_date'].head()
    return


@app.cell
def _(df, pd):
    # Convert numeric columns to appropriate data types
    numeric_cols = [
        'monthly_spend',
        'weekly_purchases',
        'average_order_value',
        'age',
        'income_level',
        'household_size',
        'daily_session_time_minutes',
        'product_views_per_day',
        'app_usage_frequency',
        'wishlist_items_count',
        'engagement_score',
        'ad_views_per_day',
        'ad_clicks_per_day',
        'notification_response_rate',
        'coupon_usage_frequency',
        'ctr',
        'coupon_dependency',
        'cart_abandonment_rate',
        'checkout_abandonments_per_month',
        'return_rate',
        'return_frequency',
        'risk_score',
        'loyalty_program_member',
        'premium_subscription',
        'account_age_months',
        'purchase_month',
        'purchase_day',
        'revenue_proxy',
        'lifetime_value_proxy',
        'risk_adjusted_value',
        'spend_per_purchase',
        'spend_to_aov_ratio',
        'spend_per_view',
        'engagement_spend',
        'risk_spend'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return


@app.cell
def _(df):
    print(df.columns)
    return


@app.cell
def _(df):
    # Check Columns
    df['loyalty_program_member'].value_counts()
    return


@app.cell
def _(df):
    # Check Columns
    df['premium_subscription'].value_counts()
    return


@app.cell
def _(df):
    # Check Columns
    df['loyalty_program_member'].value_counts()
    return


@app.cell
def _(df):
    # Check Columns
    df['premium_subscription'].value_counts()
    return


@app.cell
def _(df, np, pd):
    # Make New Columns for Loyalty and Premium Subscription

    # Normalize binary columns 
    def normalize_binary(series):
        s = series.copy()
        s_num = pd.to_numeric(s, errors='coerce')
        if s_num.dropna().isin([0, 1]).all():
            return s_num
        return s.astype(str).str.strip().str.lower().map({
            'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0
        })

    # Create new binary flags and labels
    df['loyalty_program_member_flag'] = normalize_binary(df['loyalty_program_member'])
    df['premium_subscription_flag'] = normalize_binary(df['premium_subscription'])

    df['loyalty_program_member_label'] = df['loyalty_program_member_flag'].map({
        1: 'Loyalty Member',
        0: 'Non-Loyalty'
    })
    df['premium_subscription_label'] = df['premium_subscription_flag'].map({
        1: 'Premium',
        0: 'Non-Premium'
    })

    df['customer_group'] = np.select(
        [
            (df['loyalty_program_member_flag'] == 1) & (df['premium_subscription_flag'] == 1),
            (df['loyalty_program_member_flag'] == 1) & (df['premium_subscription_flag'] == 0),
            (df['loyalty_program_member_flag'] == 0) & (df['premium_subscription_flag'] == 1),
            (df['loyalty_program_member_flag'] == 0) & (df['premium_subscription_flag'] == 0),
        ],
        [
            'Loyalty + Premium',
            'Loyalty Only',
            'Premium Only',
            'Standard'
        ],
        default='Unknown'
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Missing last_purchase_date: {df['last_purchase_date'].isna().sum()}")
    return


@app.cell
def _(df):
    # Display the new columns
    df[[
        'loyalty_program_member_flag', 'premium_subscription_flag',
        'loyalty_program_member_label', 'premium_subscription_label',
        'customer_group'
    ]].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The data is suitable for use for business analysis where there are no missing values ​​and creating new Loyalty and Premium Subscription columns.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Executive Descriptive Summary**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Business question:**
    What does the customer base look like across value, activity, marketing, risk, and recency dimensions?

    ---
    """)
    return


@app.cell
def _(df, display, np):
    # Create a summary table for numeric columns
    summary_cols = [
        'monthly_spend', 'weekly_purchases', 'average_order_value',
        'daily_session_time_minutes', 'product_views_per_day', 'engagement_score',
        'ctr', 'notification_response_rate', 'coupon_usage_frequency', 'coupon_dependency',
        'cart_abandonment_rate', 'return_rate', 'risk_score',
        'account_age_months', 'lifetime_value_proxy', 'risk_adjusted_value'
    ]

    summary_table = df[summary_cols].agg(['count', 'mean', 'median', 'std', 'min', 'max']).T
    summary_table['missing_pct'] = df[summary_cols].isna().mean() * 100
    summary_table['cv'] = summary_table['std'] / summary_table['mean'].replace(0, np.nan)

    display(summary_table.sort_values('mean', ascending=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This question is asking for a general profile of the customers from several angles:

    - Value = how valuable customers are to the business
    - Activity = how active customers are
    - Marketing = how customers respond to promotions and marketing
    - Risk = how risky the customers are
    - Recency = how recent or how long customers have been with the business

    ---

    **1. Value dimension**

    Customers appear to have good business value overall, but the value is not evenly distributed.

    - Risk adjusted value:
        - mean = 71,524,
        - median = 45,551
    - Lifetime value proxy:
        - mean = 31,256,
        - median = 23,616
    - Monthly spend:
        - mean = 2,500
    - Average order value:
        - mean = 255

    What this means:

    - Customers are generally valuable.
    - But since the mean is higher than the median, a smaller group of high-value customers is pulling the average upward.
    - This suggests the customer base has:
        - many mid-value customers
        - some very high-value customers

    ---

    **2. Activity dimension**

    The customer base looks quite active.

    - Daily session time: about 60 minutes per day
    - Product views per day: about 25.5
    - Weekly purchases: about 10.5
    - Engagement score: about 0.50

    What this means:

    - Customers spend a fair amount of time on the platform.
    - They browse products often and purchase regularly.
    - Overall, engagement looks moderate to strong, not low.

    ---

    **3. Marketing dimension**

    Customers show moderate response to marketing, but behavior varies across customers.

    - Notification response rate: about 50%
    - CTR:
        - mean = 0.433,
        - median = 0.222
    - Coupon usage frequency: about 2
    - Coupon dependency: very low at 0.0037
    - Cart abandonment rate: about 40.2%

    What this means:

    - Customers respond to notifications at a moderate level.
    - CTR is uneven: many customers click less, while some click much more.
    - Customers do use coupons, but they are not highly dependent on them.
    - Cart abandonment is still fairly high, so many customers add items to cart but do not complete the purchase.

    ---

    **4. Risk dimension**

    The overall customer base seems to have moderate risk.

    - Risk score:
        - mean = 0.479
        - median = 0.478

    What this means:

    - Most customers are in the middle risk range.
    - Since the mean and median are very close, risk seems fairly balanced across the customer base.
    - There is no strong sign that most customers are either extremely low-risk or extremely high-risk.

    ---

    **5. Recency dimension**

    The closest variable here is account age.

    Account age:
        - mean = 12.5 months
        - median = 13 months
        - range = 1 to 24 months

    What this means:

    The customer base is relatively young, with the average customer being around 1 year old.
    Most customers seem to have joined within the last 1–2 years.
    However, this shows customer tenure, not true recency like “days since last purchase.”

    ---

    **Overall conclusion**

    In simple terms, the customer base looks like this:

    - The customer base is large, fairly active, and reasonably valuable overall.
    - Most customers fall into the middle range, but a smaller group of high-value customers raises the average value.
    - Customers are fairly engaged and moderately responsive to marketing. Risk levels are mostly moderate, and the customer base is relatively young, with an average account age of around one year.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Customer Profile Analysis**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Business question:**
    Which demographic and socioeconomic groups appear more commercially valuable, more active, or more exposed to risk?

    ---
    """)
    return


@app.cell
def _(df, display):
    # Create business profiles based on key dimensions
    profile_dims = ['gender', 'country', 'income_group', 'employment_status', 'education_level']

    for dim in profile_dims:
        profile_kpi = (
            df.groupby(dim, dropna=False)
            .agg(
                customers=('monthly_spend', 'size'),
                avg_monthly_spend=('monthly_spend', 'mean'),
                avg_weekly_purchases=('weekly_purchases', 'mean'),
                avg_ltv=('lifetime_value_proxy', 'mean'),
                avg_risk=('risk_score', 'mean'),
                avg_engagement=('engagement_score', 'mean')
            )
            .sort_values('avg_ltv', ascending=False)
        )

        display(profile_kpi)
    return


@app.cell
def _(df, display, plt):
    # Analyze top countries by average lifetime value proxy
    top_countries = df['country'].value_counts().head(10).index
    country_subset = df[df['country'].isin(top_countries)]

    country_perf = (
        country_subset.groupby('country')
        .agg(
            customers=('monthly_spend', 'size'),
            avg_monthly_spend=('monthly_spend', 'mean'),
            avg_weekly_purchases=('weekly_purchases', 'mean'),
            avg_ltv=('lifetime_value_proxy', 'mean'),
            avg_risk=('risk_score', 'mean')
        )
        .sort_values('avg_ltv', ascending=False)
    )

    display(country_perf)

    plt.figure(figsize=(12, 5))
    country_perf['avg_ltv'].sort_values().plot(kind='barh')
    plt.title('Average Lifetime Value Proxy by Top Country')
    plt.xlabel('Average Lifetime Value Proxy')
    plt.ylabel('Country')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    It compares groups such as:

    - Gender
    - Country
    - Income group
    - Employment status
    - Education level

    Using:

    - Commercial value → average LTV and monthly spend
    - Activity → weekly purchases and engagement
    - Risk → average risk score
    - Simple answer based on the data

    Overall, the main finding is:

    - The differences between groups are very small.
    - No group stands out as dramatically better or worse than the others.

    ---

    1. Gender

        **Gender does not make a big difference.**

    - Female customers spend slightly more and purchase slightly more often.
    - Male customers have slightly higher LTV and engagement.
    - Other has slightly higher risk.

        But these gaps are very small.

    ---

    2. Country

        **Country shows some small differences.**

    - India looks slightly strongest in business value, with the highest LTV, monthly spend, and weekly purchases.
    - USA has the highest engagement.
    - Brazil appears slightly more risky.
    - UK appears slightly less risky.

        Still, the differences across countries are minor.

    ---

    3. Income group

        **Income group gives a slightly clearer pattern.**

    - Upper-Middle customers look the most attractive overall: highest LTV, highest purchase frequency, and relatively low risk.
    - Very Low income customers appear slightly more risky and have the lowest LTV.
    - Middle income customers show the highest engagement.

        So, Upper-Middle looks like the strongest income segment.

    ---

    4. Employment status

        **Differences are also small here.**

    - Students have the highest LTV.
    - Retired customers purchase slightly more often.
    - Self-employed customers spend the most monthly, but also have the highest risk.
    - Employed customers have the lowest risk.

    ---

    5. Education level

        **Education shows a small pattern as well.**

    - PhD customers have the highest LTV and weekly purchases.
    - Master customers are also strong in value.
    - Associate Degree has the highest engagement.
    - PhD also has the highest risk, while Bachelor has the lowest.

    ---

    **Final conclusion:**

    Some groups look slightly stronger than others:

    - India in value and activity
    - Upper-Middle income in overall attractiveness
    - Students and Retired in value or activity
    - PhD in value and activity, but also in risk

    However, the most important takeaway is:

    Customer behavior is broadly similar across demographic and socioeconomic groups.
    So, these factors help explain some variation, but they are not strong differentiators on their own

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Transaction Intensity Analysis**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Business question:**
    Which customers buy more frequently, and how does shopping intensity relate to value creation?

    ---
    """)
    return


@app.cell
def _(df, display, pd):
    # Create purchase frequency segments based on weekly purchases
    df['purchase_frequency_segment'] = pd.qcut(
        df['weekly_purchases'].rank(method='first'),
        q=4,
        labels=['Low Frequency', 'Mid-Low Frequency', 'Mid-High Frequency', 'High Frequency']
    )

    # Analyze KPIs by purchase frequency segments
    frequency_kpi = (
        df.groupby('purchase_frequency_segment')
        .agg(
            customers=('weekly_purchases', 'size'),
            avg_weekly_purchases=('weekly_purchases', 'mean'),
            avg_monthly_spend=('monthly_spend', 'mean'),
            avg_aov=('average_order_value', 'mean'),
            avg_spend_per_purchase=('spend_per_purchase', 'mean'),
            avg_ltv=('lifetime_value_proxy', 'mean'),
            avg_risk=('risk_score', 'mean'),
        )
        .sort_values('avg_weekly_purchases', ascending=False)
    )

    display(frequency_kpi)
    return


@app.cell
def _(df, display):
    # Analyze shopping habits based on session time, product views, wishlist items, and engagement score
    shopping_habit_kpi = (
        df.groupby('purchase_frequency_segment')
        .agg(
            avg_session_time=('daily_session_time_minutes', 'mean'),
            avg_product_views=('product_views_per_day', 'mean'),
            avg_wishlist_items=('wishlist_items_count', 'mean'),
            avg_engagement_score=('engagement_score', 'mean')
        )
    )

    display(shopping_habit_kpi)
    return


@app.cell
def _(df, plt, sns):
    # Visualize the relationship between weekly purchases and monthly spend
    agg_mean = (
        df.groupby('weekly_purchases', as_index=False)['monthly_spend']
          .mean()
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=agg_mean, x='weekly_purchases', y='monthly_spend', marker='o')
    plt.title('Average Monthly Spend per Weekly Purchases')
    plt.xlabel('Weekly Purchases')
    plt.ylabel('Average Monthly Spend')
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Visualize the relationship between weekly purchases and monthly spend using median
    agg_median = (
        df.groupby('weekly_purchases', as_index=False)['monthly_spend']
          .median()
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=agg_median, x='weekly_purchases', y='monthly_spend', marker='o')
    plt.title('Median Monthly Spend per Weekly Purchases')
    plt.xlabel('Weekly Purchases')
    plt.ylabel('Median Monthly Spend')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    **Which customers buy most often?**

    This is measured by:

    * **avg_weekly_purchases**

    **Does higher shopping intensity create more value?**

    This is measured by:

    * **avg_monthly_spend**
    * **avg_ltv**
    * **avg_aov**
    * **avg_spend_per_purchase**
    * sometimes also supported by **risk** and **engagement**

    ---

    **The customers who buy most frequently are the **High Frequency** segment**

    The ranking is clear:

    * **High Frequency** = **17.99** purchases per week
    * **Mid-High Frequency** = **12.99**
    * **Mid-Low Frequency** = **7.99**
    * **Low Frequency** = **2.99**

    So, if the question is **which customers buy more often**, the answer is:

    **High Frequency customers.**

    ---

    **But buying more often does not make total customer value much higher**

    This is the most important insight.

    Monthly spend is almost the same across all groups:

    * **Low Frequency** = **2502.41**
    * **Mid-Low Frequency** = **2499.52**
    * **High Frequency** = **2498.41**
    * **Mid-High Frequency** = **2497.72**

    This means:

    **Even though some customers buy much more often, their total monthly spending is almost the same as customers who buy less often.**

    So:

    * frequent buyers make more transactions
    * low-frequency buyers make fewer transactions, but each transaction is larger

    ---

    **3. Customer value (LTV) is also almost the same**

    Average LTV:

    * **Low Frequency** = **31307.21** → highest
    * **Mid-Low Frequency** = **31254.76**
    * **High Frequency** = **31236.02**
    * **Mid-High Frequency** = **31225.75**

    This means:

    **The customers who buy most often do not have the highest LTV.**
    In fact, the **Low Frequency** segment is slightly higher.

    But the differences are very small, so the safest conclusion is:

    **Purchase frequency does not show a strong relationship with higher customer value in this data.**

    ---

    **4. What really changes is the **transaction pattern****

    Look at **avg_spend_per_purchase**:

    * **Low Frequency** = **1144.12**
    * **Mid-Low Frequency** = **323.35**
    * **Mid-High Frequency** = **194.62**
    * **High Frequency** = **139.70**

    This is very important.

    It means:

    * **Low Frequency customers** buy **less often**, but spend **much more each time**
    * **High Frequency customers** buy **more often**, but spend **less per purchase**

    So the main relationship is:

    **The more frequently customers buy, the smaller their spend per purchase tends to be.**
    And the less frequently they buy, the larger each purchase tends to be.

    ---

    **5. Average Order Value is also very similar**

    The **avg_aov** stays around **254–255** for all groups.

    This means:

    * order value is very similar across segments
    * higher purchase frequency does not come with much higher order value

    ---

    **6. Engagement is almost the same too**

    From the second table:

    * **avg_session_time** is about **60 minutes** for all groups
    * **avg_product_views** is about **25.5**
    * **avg_wishlist_items** is about **10**
    * **avg_engagement_score** is about **0.502**

    This means:

    **Customers who buy more often are not much more active in browsing or engagement than other groups.**

    So higher shopping frequency does **not necessarily mean**:

    * spending more time on the platform
    * viewing many more products
    * having much higher engagement

    ---

    **7. Risk is also very similar**

    Average risk:

    * **Low Frequency** = **0.47833** → lowest
    * **Mid-Low Frequency** = **0.47930** → highest
    * the others are very close

    This means:

    **Risk differences across purchase frequency segments are also minimal.**

    ---

    **What do the charts show?**

    In both the **Average Monthly Spend per Weekly Purchases** and **Median Monthly Spend per Weekly Purchases** charts, the line moves up and down only slightly around a very similar level.

    This means:

    **As weekly purchases increase, monthly spend does not increase in a clear way.**

    So the charts support the same conclusion:

    **Buying more often does not automatically lead to higher total monthly spending.**

    ---

    **Final conclusion**

    * The customers who buy most frequently are the **High Frequency** segment.
    * However, **shopping intensity does not have a strong relationship with value creation**.
    * **Monthly spend and LTV are almost the same across all segments.**
    * What really changes is the **spending pattern**:

      * **High Frequency** customers buy often, but spend less per purchase
      * **Low Frequency** customers buy less often, but spend much more each time

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Marketing Responsiveness Analysis**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Business question:**
    Which customers respond to campaigns, and does that responsiveness translate into stronger commercial outcomes?

    ---
    """)
    return


@app.cell
def _(df, display, pd):
    # Create marketing response segments based on click-through rate (CTR)
    df['marketing_response_segment'] = pd.qcut(
        df['ctr'].rank(method='first'),
        q=3,
        labels=['Low Response', 'Medium Response', 'High Response']
    )

    # Analyze KPIs by marketing response segments
    marketing_kpi = (
        df.groupby('marketing_response_segment')
        .agg(
            customers=('ctr', 'size'),
            avg_ctr=('ctr', 'mean'),
            avg_ad_views=('ad_views_per_day', 'mean'),
            avg_ad_clicks=('ad_clicks_per_day', 'mean'),
            avg_notification_response=('notification_response_rate', 'mean'),
            avg_coupon_usage=('coupon_usage_frequency', 'mean'),
            avg_coupon_dependency=('coupon_dependency', 'mean'),
            avg_monthly_spend=('monthly_spend', 'mean'),
            avg_weekly_purchases=('weekly_purchases', 'mean'),
            avg_ltv=('lifetime_value_proxy', 'mean'),
            avg_risk=('risk_score', 'mean')
        )
        .sort_values('avg_ctr', ascending=False)
    )

    display(marketing_kpi)
    return (marketing_kpi,)


@app.cell
def _(df, display, pd, plt, sns):
    # Create coupon dependency segments
    df['coupon_dependency_segment'] = pd.qcut(
        df['coupon_dependency'].rank(method='first'),
        q=4,
        labels=['Low Coupon Dependency', 'Mid-Low', 'Mid-High', 'High Coupon Dependency']
    )

    # Analyze KPIs by coupon dependency segments
    coupon_kpi = (
        df.groupby('coupon_dependency_segment')
        .agg(
            customers=('coupon_dependency', 'size'),
            avg_monthly_spend=('monthly_spend', 'mean'),
            avg_weekly_purchases=('weekly_purchases', 'mean'),
            avg_aov=('average_order_value', 'mean'),
            avg_ltv=('lifetime_value_proxy', 'mean'),
            avg_risk=('risk_score', 'mean')
        )
        .sort_values('avg_ltv', ascending=False)
    )

    display(coupon_kpi)

    # Visualize the relationship between marketing response segments and monthly spend
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='marketing_response_segment', y='monthly_spend')
    plt.title('Monthly Spend by Marketing Response Segment')
    plt.xlabel('Marketing Response Segment')
    plt.ylabel('Monthly Spend')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    **Which customers respond most to campaigns?**

    This is usually measured by:

    * **CTR**
    * **ad clicks**
    * **notification response**
    * **coupon usage**
    * **coupon dependency**

    **2. Do more responsive customers perform better commercially?**

    This is usually measured by:

    * **monthly spend**
    * **weekly purchases**
    * **LTV**
    * **risk**

    ---

    **1. The most responsive customers are in the **High Response** segment**

    From the table:

    * **High Response**

      * CTR = **1.0268** → highest
      * Ad clicks = **3.712** → highest
    * **Medium Response**

      * CTR = **0.2287**
      * Ad clicks = **3.114**
    * **Low Response**

      * CTR = **0.0437**
      * Ad clicks = **0.665**

    This means:

    **High Response customers are the most responsive to ads and campaigns.**

    An interesting point is that the **High Response** group actually has **fewer ad views** than the Medium and Low Response groups, but they still generate more clicks.
    That suggests they are **more efficient in responding to campaigns**.

    ---

    **2. But higher campaign responsiveness does not automatically lead to stronger business outcomes**

    This is the most important insight.

    **Monthly spend**

    * **Medium Response** = **2501.91** → highest
    * **Low Response** = **2498.95**
    * **High Response** = **2497.67** → lowest

    **Weekly purchases**

    * **Medium Response** = **10.504** → highest
    * **High Response** = **10.487**
    * **Low Response** = **10.485**

    **LTV**

    * **Medium Response** = **31268.48** → highest
    * **Low Response** = **31255.10**
    * **High Response** = **31244.22**

    This means:

    **The customers who respond most strongly to campaigns are not the ones with the highest spend, purchase frequency, or LTV.**

    In fact, **Medium Response** performs slightly better on business outcomes.

    ---

    **3. The chart supports the same conclusion**

    In the **Monthly Spend by Marketing Response Segment** boxplot, the monthly spend distributions for:

    * **Low Response**
    * **Medium Response**
    * **High Response**

    look **almost identical**.

    This means:

    **Even though campaign responsiveness differs, monthly spending remains very similar across the segments.**

    So visually, the chart also shows that:

    **higher campaign response does not create a large difference in spending.**

    ---

    **4. Notification response and coupon usage are also very similar**

    From the table:

    **Notification response**

    * High = **49.98**
    * Medium = **49.97**
    * Low = **50.04**

    **Coupon usage**

    * High = **2.003**
    * Medium = **2.003**
    * Low = **2.000**

    This means:

    **The main differences between the segments come from CTR and ad clicks, not from notification response or coupon usage.**

    For those two metrics, the segments are almost the same.

    ---

    **5. Risk is also very similar**

    * **Medium Response** = **0.47857** → lowest
    * **High Response** = **0.47867**
    * **Low Response** = **0.47899** → highest

    The gap is very small.

    This means:

    **Higher campaign responsiveness is not strongly linked to much lower or much higher risk either.**

    ---

    **Additional insight from coupon dependency**

    The **coupon dependency** table gives a much stronger business signal.

    **Results:**

    * **Mid-Low coupon dependency**

      * monthly spend = **3375.78**
      * LTV = **42212.40**
      * risk = **0.47846**

    * **Low coupon dependency**

      * monthly spend = **2873.70**
      * LTV = **35944.24**

    * **Mid-High coupon dependency**

      * monthly spend = **2812.30**
      * LTV = **35140.25**

    * **High coupon dependency**

      * monthly spend = **936.27**
      * LTV = **11726.86**
      * risk = **0.47922** → highest

    **What this means:**

    Customers who are **highly dependent on coupons** tend to have:

    * much lower spend
    * much lower LTV
    * slightly higher risk

    Meanwhile, customers with **low or mid-low coupon dependency** generate much stronger value.

    So the deeper insight is:

    **It is not just about whether customers respond to campaigns, but how they respond.**

    If customers respond mainly because of coupons and discounts, the business outcome may actually be weaker.

    ---

    **Final conclusion**

    * The customers who respond most to campaigns are the **High Response** segment.
    * But **higher campaign responsiveness does not automatically lead to stronger commercial outcomes**.
    * **Monthly spend, weekly purchases, and LTV are very similar** across the response segments.
    * In fact, **Medium Response** performs slightly better than High Response on business value.
    * The stronger insight comes from **coupon dependency**:

      * the higher the coupon dependency, the lower the customer value
      * customers with **low or mid-low coupon dependency** appear much more valuable

    **Campaign responsiveness shows who is easy to engage with marketing, but it does not always mean they create more business value. What matters more is whether they depend heavily on discounts or not.**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Customer Risk Signal Analysis**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Business question:**
    Which customers show signals of revenue leakage or unstable commercial value?

    ---
    """)
    return


@app.cell
def _(df, display, pd):
    # Create risk bands based on risk score
    df['risk_band'] = pd.qcut(
        df['risk_score'].rank(method='first'),
        q=4,
        labels=['Low Risk', 'Moderate Risk', 'Elevated Risk', 'High Risk']
    )

    # Analyze KPIs by risk bands
    risk_kpi = (
        df.groupby('risk_band')
        .agg(
            customers=('risk_score', 'size'),
            avg_risk_score=('risk_score', 'mean'),
            avg_cart_abandonment=('cart_abandonment_rate', 'mean'),
            avg_checkout_abandonment=('checkout_abandonments_per_month', 'mean'),
            avg_return_rate=('return_rate', 'mean'),
            avg_return_frequency=('return_frequency', 'mean'),
            avg_monthly_spend=('monthly_spend', 'mean'),
            avg_ltv=('lifetime_value_proxy', 'mean'),
            avg_risk_adjusted_value=('risk_adjusted_value', 'mean'),
            avg_engagement_score=('engagement_score', 'mean')
        )
        .sort_values('avg_risk_score', ascending=False)
    )

    display(risk_kpi)
    return (risk_kpi,)


@app.cell
def _(df, display, pd):
    # Create value bands based on lifetime value proxy
    df['value_band_3'] = pd.qcut(
        df['lifetime_value_proxy'].rank(method='first'),
        q=3,
        labels=['Low Value', 'Mid Value', 'High Value']
    )

    value_risk_matrix = (
        df.groupby(['value_band_3', 'risk_band'])
        .agg(
            customers=('monthly_spend', 'size'),
            avg_monthly_spend=('monthly_spend', 'mean'),
            avg_ltv=('lifetime_value_proxy', 'mean'),
            avg_risk_adjusted_value=('risk_adjusted_value', 'mean'),
            avg_risk_score=('risk_score', 'mean')
        )
    )

    display(value_risk_matrix)
    return


@app.cell
def _(df, plt, sns):
    # Visualize the value-risk matrix 
    value_cut = df['lifetime_value_proxy'].median()
    risk_cut = df['risk_score'].median()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x='lifetime_value_proxy',
        y='risk_score',
        alpha=0.3,
        ax=ax
    )

    ax.axvline(value_cut, linestyle='--')
    ax.axhline(risk_cut, linestyle='--')

    # Ambil batas area plot
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Posisi label di tengah masing-masing kuadran
    label_positions = {
        'Low Value / High Risk': ((xmin + value_cut) / 2, (risk_cut + ymax) / 2),
        'High Value / High Risk': ((value_cut + xmax) / 2, (risk_cut + ymax) / 2),
        'Low Value / Low Risk': ((xmin + value_cut) / 2, (ymin + risk_cut) / 2),
        'High Value / Low Risk': ((value_cut + xmax) / 2, (ymin + risk_cut) / 2),
    }

    for label, (x, y) in label_positions.items():
        ax.text(
            x, y, label,
            ha='center',
            va='center',
            fontsize=11,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
        )

    ax.set_title('Customer Quadrant: Value vs Risk')
    ax.set_xlabel('Lifetime Value Proxy')
    ax.set_ylabel('Risk Score')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Typical warning signs are:

    * **high cart abandonment** → they often leave before completing a purchase
    * **high checkout abandonment** → they drop off during checkout
    * **high return rate** → more revenue comes back or gets reversed
    * **high risk score** → their value is less stable
    * **risk-adjusted value much lower than LTV** → they may look valuable on paper, but their true value drops once risk is considered

    ---


    The customers showing the clearest signs of **revenue leakage** or **unstable commercial value** are:

    * **High Risk customers**
    * especially the **High Value + High Risk** group
    * followed by **Mid Value + High Risk**
    * and, to a lesser extent, **Elevated Risk** customers

    ---

    **1. High Risk customers are the clearest warning signal**

    From the **risk_band** table:

    **High Risk**

    * **avg_cart_abandonment = 62.79** → highest
    * **avg_return_rate = 69.16** → highest
    * **avg_monthly_spend = 2495.66**
    * **avg_ltv = 31238.34**
    * **avg_risk_adjusted_value = 24930.46** → by far the lowest

    **Compared with Low Risk**

    * **cart abandonment**: 62.79 vs 18.08
    * **return rate**: 69.16 vs 30.63
    * **LTV**: almost the same, 31238 vs 31261
    * **risk-adjusted value**: 24930 vs 129037

    **This is very important:**

    **At the top-line level, High Risk customers appear almost as valuable as Low Risk customers, because monthly spend and LTV look similar.**
    But once risk is taken into account, their value drops sharply.

    So this group is the clearest example of:

    **“customers who appear valuable, but whose value is leaking or unstable.”**

    ---

    **2. Checkout abandonment is not the main issue in this data**

    Look at **avg_checkout_abandonment**:

    * High Risk = **5.00**
    * Elevated Risk = **4.99**
    * Moderate Risk = **5.00**
    * Low Risk = **5.00**

    These values are almost identical.

    **What this means**

    **The main problem is not checkout abandonment**, because this metric barely changes across risk bands.

    So the stronger signs of leakage here come from:

    * **cart abandonment**
    * **return rate**
    * **the drop in risk-adjusted value**

    ---

    **3. Return rate is a very strong leakage signal**

    From the table:

    * **Low Risk** = **30.63**
    * **Moderate Risk** = **45.24**
    * **Elevated Risk** = **55.03**
    * **High Risk** = **69.16**

    **What this means**

    As risk increases, return rate also increases.

    This suggests:

    * more revenue is being reversed or lost
    * customer value becomes less stable
    * gross revenue does not necessarily turn into secure net value

    ---

    **4. Risk-adjusted value shows the clearest gap**

    This is the most important metric for answering the question.

    **Risk-adjusted value by risk band**

    * **Low Risk** = **129036.77**
    * **Moderate Risk** = **79616.95**
    * **Elevated Risk** = **52511.79**
    * **High Risk** = **24930.46**

    **What this means**

    Even though **monthly spend** and **LTV** stay almost the same across risk bands, value drops dramatically once risk is considered.

    So:

    **The higher the risk, the more likely the customer’s value is unstable or leaking.**

    ---

    **Stronger insight from the value + risk combination**

    The **value_band_3 + risk_band** table gives a deeper answer.

    ---

    **5. The most critical group: **High Value + High Risk****

    This is the segment that deserves the most attention.

    **Data:**

    * **avg_monthly_spend = 3640.49**
    * **avg_ltv = 63604.69**
    * **avg_risk_adjusted_value = 36374.96**
    * **avg_risk_score = 0.6939**

    **Compared with **High Value + Low Risk**:**

    * monthly spend is almost the same:

      * **3640.49 vs 3647.86**
    * LTV is also almost the same:

      * **63604.69 vs 63730.54**
    * but risk-adjusted value is:

      * **36374.96 vs 188123.63**

    **What this means**

    This is a very strong signal of **unstable value**.

    This group:

    * looks very valuable in spend and LTV
    * but once risk is considered, its value drops heavily

    So **High Value + High Risk** is the segment with:

    * **high revenue potential**
    * but also the **largest risk of value leakage**

    This is usually the top priority for:

    * retention control
    * return reduction
    * fraud or risk monitoring
    * revenue quality review

    ---

    **6. The second group to watch: **Mid Value + High Risk** **

    **Data:**

    * **avg_monthly_spend = 2459.81**
    * **avg_ltv = 24272.21**
    * **avg_risk_adjusted_value = 24594.39**

    **Compared with **Mid Value + Low Risk**:**

    * monthly spend is almost the same
    * LTV is almost the same
    * but risk-adjusted value falls from **127251.32** to **24594.39**

    **What this means**

    This group shows the same pattern:

    * the top-line numbers still look fine
    * but the quality and stability of that value are much weaker

    So this segment also shows **serious revenue leakage risk**.

    ---

    **7. Low Value + High Risk is also problematic, but less critical**

    **Data:**

    * **avg_monthly_spend = 1388.42**
    * **avg_ltv = 5882.61**
    * **avg_risk_adjusted_value = 13839.54**

    **What this means**

    This group is clearly risky, but it already starts from a lower value base.

    So:

    * it still shows leakage signals
    * but from a business perspective, it is usually **less urgent** than **High Value + High Risk**

    Because the most damaging situation is usually:

    **customers with high apparent value whose value is not secure.**

    ---

    **8. Elevated Risk customers are also worth monitoring**

    The **Elevated Risk** group sits in the middle:

    * cart abandonment is already high
    * return rate is already high
    * risk-adjusted value has already fallen a lot compared with Low Risk

    **What this means**

    This can be seen as an **early warning segment**.

    They are not as severe as High Risk customers yet, but they already show signs that:

    * value is starting to leak
    * revenue quality is weakening
    * commercial risk is increasing

    ---

    **9. Engagement is not the main differentiator**

    Look at **avg_engagement_score**:

    * High Risk = **0.5015**
    * Elevated Risk = **0.5023**
    * Moderate Risk = **0.5021**
    * Low Risk = **0.5020**

    These are almost identical.

    **What this means**

    The leakage problem here is **not driven by lower engagement**.
    These customers may still be active, but they:

    * abandon more
    * return more
    * carry more risk
    * and therefore create less stable value

    ---

    **What does the quadrant chart show?**

    In the **Customer Quadrant: Value vs Risk** chart, there are four zones:

    * **Low Value / Low Risk**
    * **Low Value / High Risk**
    * **High Value / Low Risk**
    * **High Value / High Risk**

    **Key insight from the chart:**

    The most important segment for this question is:

    **High Value / High Risk**

    Because these customers:

    * have high revenue potential
    * but also high risk
    * so they are the most likely to show **revenue leakage** or **unstable commercial value**

    Meanwhile:

    **Low Value / High Risk**

    is also problematic, but the business impact is usually smaller because the base value is already low.

    ---

    **Final conclusion**

    The customers showing the clearest signs of **revenue leakage** or **unstable commercial value** are:

    * **High Risk customers**
    * especially **High Value + High Risk**
    * followed by **Mid Value + High Risk**
    * with **Elevated Risk** acting as an early warning group

    Because they show:

    * **high cart abandonment**
    * **high return rates**
    * **much lower risk-adjusted value**
    * even though **monthly spend and LTV still look similar** to safer groups

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Strategic Customer Group Comparison**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Business question:**
    Do loyalty and premium programs align with better customer economics and stronger behavioral quality?

    ---
    """)
    return


@app.cell
def _(df, display):
    # Analyze KPIs by customer groups
    group_kpi = (
        df.groupby('customer_group')
        .agg(
            customers=('monthly_spend', 'size'),
            avg_monthly_spend=('monthly_spend', 'mean'),
            avg_weekly_purchases=('weekly_purchases', 'mean'),
            avg_aov=('average_order_value', 'mean'),
            avg_engagement=('engagement_score', 'mean'),
            avg_ctr=('ctr', 'mean'),
            avg_coupon_dependency=('coupon_dependency', 'mean'),
            avg_risk=('risk_score', 'mean'),
            avg_ltv=('lifetime_value_proxy', 'mean'),
            avg_risk_adjusted_value=('risk_adjusted_value', 'mean')
        )
        .sort_values('avg_ltv', ascending=False)
    )

    display(group_kpi)
    return (group_kpi,)


@app.cell
def _(group_kpi, plt, sns):
    # Visualize the KPI profile of customer groups using a heatmap
    kpi_cols = [
        'customers',
        'avg_monthly_spend',
        'avg_weekly_purchases',
        'avg_aov',
        'avg_engagement',
        'avg_ctr',
        'avg_coupon_dependency',
        'avg_risk',
        'avg_ltv',
        'avg_risk_adjusted_value'
    ]

    heatmap_df = group_kpi[kpi_cols].copy()

    # Normalize the data for better heatmap visualization
    heatmap_norm = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min())

    # Create a label map for better readability in the heatmap
    label_map = {
        'customers': 'Customers',
        'avg_monthly_spend': 'Avg Monthly\nSpend',
        'avg_weekly_purchases': 'Avg Weekly\nPurchases',
        'avg_aov': 'Avg AOV',
        'avg_engagement': 'Avg\nEngagement',
        'avg_ctr': 'Avg CTR',
        'avg_coupon_dependency': 'Coupon\nDependency',
        'avg_risk': 'Avg Risk',
        'avg_ltv': 'Avg LTV',
        'avg_risk_adjusted_value': 'Risk Adjusted\nValue'
    }

    plot_df = heatmap_norm.rename(columns=label_map)

    plt.figure(figsize=(14, 6))
    ax_2 = sns.heatmap(
        plot_df,
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',
        linewidths=0.5,
        vmin=0,
        vmax=1,
        annot_kws={'size': 10},
        cbar_kws={'shrink': 0.9, 'label': 'Normalized Score'}
    )

    ax_2.set_title('Normalized KPI Profile by Strategic Customer Group', fontsize=16, pad=12)
    ax_2.set_xlabel('')
    ax_2.set_ylabel('Customer Group', fontsize=12)

    ax_2.set_xticklabels(ax_2.get_xticklabels(), rotation=0, ha='center', fontsize=11)
    ax_2.set_yticklabels(ax_2.get_yticklabels(), rotation=0, fontsize=12)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    In simple terms, it is asking whether customers in:

    * **loyalty programs**
    * **premium programs**
    * or **both**

    perform better than **standard customers**.

    It looks at two things:

    **Customer economics**:

    * average monthly spend
    * average order value
    * LTV
    * risk-adjusted value

    **Behavioral quality**:

    * engagement
    * CTR
    * coupon dependency
    * risk
    * weekly purchases

    ---

    **There is no large difference across the groups.**
    So based on this data, **loyalty and premium programs do not show a strong or consistent improvement** in customer economics or behavioral quality.

    All groups look **very similar**.

    ---

    **1. Customer economics**

    **Monthly spend**

    * **Loyalty + Premium** = **2501.13** → highest
    * **Standard** = **2500.36**
    * **Loyalty Only** = **2499.75**
    * **Premium Only** = **2495.96** → lowest

    **AOV**

    * **Loyalty + Premium** = **255.39** → highest
    * **Standard** = **255.31**
    * **Premium Only** = **254.75**
    * **Loyalty Only** = **254.66**

    **LTV**

    * **Premium Only** = **31279.94** → highest
    * **Standard** = **31263.57**
    * **Loyalty + Premium** = **31263.32**
    * **Loyalty Only** = **31230.65** → lowest

    **Risk-adjusted value**

    * **Loyalty Only** = **71564.50** → highest
    * **Loyalty + Premium** = **71539.92**
    * **Standard** = **71531.98**
    * **Premium Only** = **71421.74** → lowest

    **What this means**

    From a business value perspective:

    * **Loyalty + Premium** is slightly stronger in **monthly spend** and **AOV**
    * **Premium Only** is slightly stronger in **LTV**
    * **Loyalty Only** is slightly stronger in **risk-adjusted value**

    But the gaps are **extremely small**.

    So we cannot say:

    * loyalty is clearly much better
    * premium is clearly much more profitable
    * or loyalty + premium is clearly the strongest group

    **There is no strong, clear pattern.**

    ---

    **2. Behavioral quality**

    **Engagement**

    * **Loyalty Only** = **0.50212** → highest
    * **Loyalty + Premium** = **0.50208**
    * **Standard** = **0.50203**
    * **Premium Only** = **0.50143** → lowest

    **CTR**

    * **Premium Only** = **0.43697** → highest
    * **Loyalty + Premium** = **0.43438**
    * **Standard** = **0.43339**
    * **Loyalty Only** = **0.42977** → lowest

    **Coupon dependency**

    * **Standard** = **0.00353** → lowest, meaning least dependent on coupons
    * **Loyalty + Premium** = **0.00368**
    * **Loyalty Only** = **0.00370**
    * **Premium Only** = **0.00387** → highest

    **Risk**

    * **Loyalty + Premium** = **0.47848** → lowest
    * **Loyalty Only** = **0.47868**
    * **Premium Only** = **0.47878**
    * **Standard** = **0.47894** → highest

    **Weekly purchases**

    * **Premium Only** = **10.505** → highest
    * **Standard** = **10.501**
    * **Loyalty + Premium** = **10.487**
    * **Loyalty Only** = **10.478** → lowest

    **What this means**

    From a behavior perspective:

    * **Loyalty + Premium** looks slightly better in **risk**
    * **Loyalty Only** is slightly higher in **engagement**
    * **Premium Only** is slightly higher in **CTR** and **weekly purchases**
    * **Standard** actually has the **lowest coupon dependency**

    So again:

    **No single group clearly wins across all behavioral measures.**

    ---

    **3. What the heatmap suggests**

    The heatmap shows the same story:

    * each group is a little stronger on some KPIs
    * but no group clearly dominates most of the metrics
    * the profiles are **similar**, not dramatically different

    This means:

    **Loyalty and premium programs are associated with some variation, but the effect is small and inconsistent.**

    ---

    **Do loyalty and premium programs align with better customer economics?**

    **Slightly, but not strongly.**

    * **Loyalty + Premium** is a little better in spend and AOV
    * **Premium Only** is slightly higher in LTV
    * **Loyalty Only** is slightly higher in risk-adjusted value

    But the differences are very small, so there is **no strong evidence** that these programs create clearly better economics.

    **Do loyalty and premium programs align with stronger behavioral quality?**

    **Also not strongly or consistently.**

    * **Loyalty + Premium** has the lowest risk
    * **Loyalty Only** has the highest engagement
    * **Premium Only** has the highest CTR
    * **Standard** has the lowest coupon dependency

    So behavioral quality is also **not clearly better** for program members overall.

    ---

    **Final conclusion**

    **Loyalty and premium programs do not show a large or consistent advantage over the standard group.**
    Some program groups perform slightly better on certain metrics, but the differences are very small and not consistent across all KPIs.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Key Business Findings**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Business question:**
    What are the most important takeaways for customer strategy and future segmentation work?

    ---
    """)
    return


@app.cell
def _(df, group_kpi, marketing_kpi, risk_kpi):
    # Identify key business insights based on the analyses
    top_value_group = group_kpi['avg_ltv'].idxmax()
    top_risk_group = group_kpi['avg_risk'].idxmax()
    highest_response_segment = marketing_kpi['avg_ltv'].idxmax()
    highest_risk_band = risk_kpi['avg_risk_score'].idxmax()

    top_income_group = (
        df.groupby('income_group')
        .agg(avg_ltv=('lifetime_value_proxy', 'mean'))
        .sort_values('avg_ltv', ascending=False)
        .index[0]
    )

    top_country = (
        df.groupby('country')
        .agg(avg_ltv=('lifetime_value_proxy', 'mean'))
        .sort_values('avg_ltv', ascending=False)
        .index[0]
    )

    insights = [
        f"The highest-value strategic customer group is '{top_value_group}', based on average lifetime value proxy.",
        f"The group with the highest behavioral risk is '{top_risk_group}', which should be monitored for revenue leakage and retention issues.",
        f"Customers in the '{highest_response_segment}' marketing response segment show the strongest value potential, suggesting that responsiveness can support targeting decisions.",
        f"The most critical risk band is '{highest_risk_band}', where abandonment and return behavior are likely to reduce realized value.",
        f"The top income-based value contributor is '{top_income_group}', indicating a strong relationship between purchasing power and customer value.",
        f"The country with the highest average lifetime value proxy is '{top_country}', which may justify more localized targeting or investment."
    ]

    print("=== KEY BUSINESS FINDINGS ===")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **1. **Premium Only** is the highest-value strategic customer group**

    This means customers in the **Premium Only** group have the **highest average lifetime value (LTV)**.

    **What it means:**

    * this group is the most valuable in the long run
    * they are an important customer segment for the business
    * the company should focus on keeping them engaged and retained

    ---

    **2. **Standard** is the group with the highest behavioral risk**

    This means customers in the **Standard** group show slightly higher signs of behavioral risk, such as:

    * revenue leakage
    * churn risk
    * weaker retention
    * less stable purchasing behavior

    **What it means:**

    * this group should be monitored more closely
    * even though they are the standard segment, they may create more commercial risk
    * they may need stronger retention or risk-control strategies

    ---

    **3. **Medium Response** customers show the strongest value potential**

    This means customers in the **Medium Response** marketing segment produce the strongest value outcomes.

    **What it means:**

    * very low-response customers are less attractive
    * but the highest-response group is not necessarily the most valuable
    * the **Medium Response** group looks the most balanced and commercially promising

    **Business implication:**

    * this segment may be a better target for campaigns
    * targeting should not focus only on the most responsive customers

    ---

    **4. **High Risk** is the most critical risk band**

    This means customers in the **High Risk** group are the most likely to show:

    * high cart abandonment
    * high return behavior
    * value that looks good at first, but weakens after risk is considered

    **What it means:**

    * this is the group that needs the most attention
    * they may still show decent spend or LTV, but their real value is much less stable
    * they are a priority for risk mitigation

    ---

    **5. **Upper-Middle** is the strongest income-based value segment**

    This means the **Upper-Middle** income group contributes the highest customer value among income-based segments.

    **What it means:**

    * purchasing power has a clear relationship with customer value
    * this segment looks especially attractive for the business
    * they may be a strong target for upselling, retention, and personalization

    ---

    **6. **India** has the highest average lifetime value**

    This means customers from **India** have the highest average long-term value compared with other countries.

    **What it means:**

    * India looks like a very strong market
    * the business may benefit from:

      * more localized campaigns
      * market-specific targeting
      * greater commercial investment in India

    ---

    **Simple overall conclusion**

    In simple terms:

    * **Premium Only** = most valuable customer group
    * **Standard** = highest behavioral risk
    * **Medium Response** = strongest marketing value potential
    * **High Risk** = most critical risk segment
    * **Upper-Middle** = strongest income-based value group
    * **India** = highest-value country

    ---

    **Overall business meaning**

    These findings suggest that the business should:

    * protect and retain **Premium Only** customers
    * monitor **Standard** and **High Risk** customers more carefully
    * use **Medium Response** customers as a smart campaign target
    * prioritize **Upper-Middle** income customers
    * strengthen localized strategy in **India**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Conclusion**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **1. The customer base looks healthy, active, and reasonably valuable**

    From the overall analysis:

    * customers are fairly active in shopping and engagement
    * customer value is generally solid
    * average risk is moderate
    * most accounts are still relatively young

    **What this means:**
    The business already has a strong customer base. The main challenge is not whether valuable customers exist, but **how to protect that value and reduce value leakage**.

    **Business implication:**

    * focus more on **retention**, not only acquisition
    * improve **revenue quality**, not just transaction volume
    * prioritize managing existing customers well

    ---

    **2. Demographic and socioeconomic differences exist, but they are small**

    Across gender, country, income group, employment status, and education:

    * some groups perform slightly better than others
    * for example, **India** and **Upper-Middle income** stand out a little
    * but overall, the gaps are **small**

    **What this means:**
    Demographic and socioeconomic factors are **not the main drivers** of customer value.

    **Business implication:**

    * do not rely too heavily on demographic segmentation alone
    * use demographics as a **supporting layer**, not the main strategy
    * stronger segmentation should be based on **behavior, risk, and revenue quality**

    ---

    **3. Customers who buy more often are not necessarily more valuable**

    From the purchase frequency analysis:

    * **High Frequency** customers buy most often
    * but **monthly spend** and **LTV** are almost the same across segments
    * the real difference is in spending style:

      * frequent buyers purchase more often, but spend less each time
      * low-frequency buyers purchase less often, but spend much more each time

    **What this means:**
    Buying more often does **not automatically** mean creating more value.

    **Business implication:**

    * do not judge customer quality only by purchase frequency
    * frequent buyers and occasional buyers need different strategies
    * for low-frequency customers, focus on **basket size**
    * for high-frequency customers, focus on **efficiency and repeat experience**

    ---

    **4. High marketing response does not always mean the highest value**

    From the marketing response analysis:

    * **High Response** customers are the most responsive to campaigns
    * but **Medium Response** customers actually show slightly better business outcomes
    * so the most responsive customers are not always the most valuable

    **What this means:**
    Marketing responsiveness matters, but it is **not the same as business value**.

    **Business implication:**

    * do not target only the most responsive customers
    * **Medium Response** may be a more efficient target group
    * evaluate campaigns based on **business results**, not only clicks or CTR

    ---

    **5. Coupon dependency is a much stronger warning sign**

    This was one of the strongest findings:

    * customers with **high coupon dependency** have:

      * much lower spend
      * much lower LTV
      * slightly higher risk
    * customers with **low or mid-low coupon dependency** are much more valuable

    **What this means:**
    The real issue is not only whether customers respond to campaigns, but **whether they only respond when discounts are offered**.

    **Business implication:**

    * reduce over-reliance on heavy discounting
    * separate customers who are genuinely loyal from those who are only incentive-driven
    * use coupons more selectively, not as a default growth tool

    ---

    **6. Risk is the most important factor in protecting value**

    From the revenue leakage analysis:

    * **High Risk** is the most critical segment
    * these customers show:

      * high cart abandonment
      * high return rates
      * much lower risk-adjusted value
    * even when spend and LTV still look acceptable, their true value is much less stable

    The most concerning groups are:

    * **High Value + High Risk**
    * followed by **Mid Value + High Risk**

    **What this means:**
    Some customers look valuable on paper, but their value is actually **fragile and leaking away**.

    **Business implication:**

    * prioritize monitoring of **High Risk** customers
    * create targeted actions to:

      * reduce cart abandonment
      * reduce return rates
      * detect risky behavior earlier
    * focus on **risk-adjusted value**, not only LTV

    ---

    **7. Loyalty and premium programs do not show a strong and consistent advantage**

    From the strategic customer group analysis:

    * **Premium Only** is slightly higher in LTV
    * **Loyalty + Premium** is slightly higher in some spending metrics
    * **Loyalty Only** is slightly higher in risk-adjusted value
    * **Standard** appears to have slightly higher behavioral risk
    * but overall, the differences are still small

    **What this means:**
    Loyalty and premium programs are **not clearly creating much better customers** than the standard group.

    **Business implication:**

    * re-evaluate how effective loyalty and premium programs really are
    * measure success not just by membership size, but by:

      * spend uplift
      * retention
      * risk-adjusted value
    * if these programs are costly, make sure they produce real business impact

    ---

    **8. Some segments are still worth prioritizing**

    Even though many gaps are small, a few segments still stand out:

    * **Premium Only** → slightly strongest in LTV
    * **Medium Response** → most promising for campaign targeting
    * **High Risk** → highest priority for leakage control
    * **Upper-Middle income** → one of the strongest value segments
    * **India** → highest average LTV by country

    **What this means:**
    There are still clear priorities the business can act on.

    ---

    **Big-picture conclusion**

    In one simple sentence:

    **This business is not strongly differentiated by demographics or membership programs, but much more by customer behavior—especially risk, coupon dependency, and whether customer value is stable or leaking away.**

    ---

    **Most important business implications**

    **1. Focus on value quality, not just value size**

    Do not only track:

    * LTV
    * spend
    * transaction count

    Also track:

    * risk-adjusted value
    * return rate
    * abandonment
    * coupon dependency

    **Why?**
    Because some customers may look valuable, but a large part of that value may not be secure.

    ---

    **2. Prioritize high-risk customer management**

    The most important segments to watch are:

    * **High Risk**
    * especially **High Value + High Risk**

    **Relevant business actions:**

    * strengthen fraud and risk monitoring
    * reduce returns and abandonment
    * build alerts for high-value customers who begin showing risk signals

    ---

    **3. Use campaigns more selectively and more intelligently**

    Do not assume the most responsive customers are the best target.

    **A better approach:**

    * prioritize **Medium Response**
    * separate customers with real business potential from those who are only discount-sensitive
    * judge campaigns by business KPIs, not only engagement metrics

    ---

    **4. Reduce dependency on discounts**

    High coupon dependency is linked to lower value.

    **A healthier business direction:**

    * build loyalty that does not depend on discounts
    * use promotions for targeted activation, not as a permanent habit
    * drive value through experience, personalization, and product strength

    ---

    **5. Reassess loyalty and premium program design**

    Since the impact is not very strong:

    * test whether the programs truly improve retention and value
    * separate benefits for high-value customers versus average customers
    * redesign the programs if they add cost without clear return

    ---

    **6. Keep using priority segments for execution**

    The most useful segments for action are:

    * **Premium Only** for retention
    * **Medium Response** for campaign targeting
    * **Upper-Middle income** for upsell and personalization
    * **India** for localized market strategy
    * **High Risk** for leakage prevention

    ---
    """)
    return


if __name__ == "__main__":
    app.run()
