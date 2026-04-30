import marimo

__generated_with = "0.23.4"
app = marimo.App(auto_download=["html", "ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **E-Commerce Customer Segmentation and CRM Analytics Dashboard - Exploratory Data Analysis of Customer Behavior and Profiles**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Aim of this notebook:**
    > - Build an initial business view of customer profiles using demographic, membership, and behavioral attributes
    > - Explore the distribution of key customer variables to understand the overall customer base
    > - Create and organize additional derived fields from relevant columns to enrich customer profiling
    > - Highlight early patterns in shopping activity, engagement, and marketing-related behavior
    > - Compare customer groups at a descriptive level based on loyalty and premium subscription status
    > - Provide a structured exploratory foundation for future segmentation and deeper analysis

    > **Expected output:**
    > - A descriptive overview of customer characteristics across key business dimensions
    > - Distribution summaries for demographic, purchasing, engagement, and marketing-related variables
    > - Additional derived variables that improve customer profiling and exploratory analysis
    > - Initial comparisons between customer groups such as loyalty vs non-loyalty and premium vs non-premium
    > - Preliminary observations on customer activity, transaction behavior, and interaction patterns
    > - Early business insights that can support future segmentation, targeting, and retention analysis

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
    import os

    # Statistical tests
    from scipy import stats
    from scipy.stats import shapiro, kstest, ttest_ind, mannwhitneyu, f_oneway

    import warnings
    warnings.filterwarnings("ignore")
    return np, os, pd, plt, shapiro, sns, stats


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
    df = pd.read_csv(r"C:\Users\user\Documents\Coding\Portofolio\E-Commerce-Customer-Segmentation-and-CRM-Analytics-Dashboard\Dataset\e_commerce_shopper_enhanced.csv")
    df.head()
    return (df,)


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
    ## **General Descriptive Statistics**

    ---
    """)
    return


@app.cell
def _(df):
    # General descriptive statistics
    df.describe()
    return


@app.cell
def _(df, os, plt):
    # folder output
    output_dir = output_dir = r"C:\Users\user\Documents\Coding\Portofolio\E-Commerce-Customer-Segmentation-and-CRM-Analytics-Dashboard\Business_Analytics_Figure"
    os.makedirs(output_dir, exist_ok=True)

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    batch_size = 10

    for i in range(0, len(num_cols), batch_size):
        cols = num_cols[i:i+batch_size]
        axes = df[cols].hist(figsize=(14, 9), bins=30, edgecolor='black')

        for ax in axes.flatten():
            col = ax.get_title()
            if col:
                mean = df[col].mean()
                median = df[col].median()

                ax.axvline(mean, linestyle='--', linewidth=1)
                ax.axvline(median, linestyle=':', linewidth=1)

                ax.set_title(f"{col}\nmean={mean:.2f} | median={median:.2f}", fontsize=9)

        plt.suptitle("Distribution of Numerical Features", fontsize=14)
        plt.tight_layout()

        # Save the figure for the current batch
        filename = f"{output_dir}/hist_batch_{i//batch_size + 1}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()
    return (output_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Feature Distribution Analysis**

    ## **Summary Table**

    | Feature | Batch | Distribution | Mean | Median | Model Priority | Analysis |
    |---|---|---|---|---|---|---|
    | notification_response_rate | 1 | Uniform | 49.99 | 50.00 | Low | Perfectly flat 0–100 spread. No value concentration — likely low predictive power. Run correlation with target before including. |
    | account_age_months | 1 | Uniform | 12.51 | 13.00 | Low | Near-uniform across 0–24 months. Flat distribution suggests synthetic data. Useful only if tenure-based segmentation is needed. |
    | social_sharing_frequency | 1 | Skewed | 3.59 | 4.00 | Medium | Right-skewed, most values at 2–5. Realistic pattern — active sharers are rare. Consider log-transform before modeling. |
    | premium_subscription | 1 | Binary | 0.36 | 0.00 | High | ~64% non-premium, ~36% premium. Important binary feature likely correlated with spending behavior and basket size. |
    | return_rate | 1 | Uniform | 50.00 | 50.00 | Low | Perfectly uniform 0–100. Highly unrealistic for real data — appears to be random synthetic values with near-zero information. |
    | purchase_year | 1 | Binary | 2025.50 | 2026.00 | Medium | Only two values (2025/2026). Can serve as a temporal flag. Consider encoding as a binary indicator variable. |
    | purchase_month | 1 | Uniform | 6.52 | 7.00 | Low | Evenly distributed across all months. No seasonality detected. Consider dropping if no seasonal target is defined. |
    | purchase_day | 1 | Uniform | 15.72 | 16.00 | Low | Uniform with slight drop at end-of-month (day 31). Calendrically expected but low informational value. |
    | engagement_score | 1 | Normal | 0.50 | 0.50 | High | Bell-shaped distribution centered at 0.50 — most realistic feature in this batch. Strong summary feature, likely correlated with many behavioral signals. |
    | risk_score | 1 | Normal | 0.48 | 0.48 | High | Symmetric normal distribution. Potentially the most predictive standalone feature. Investigate its definition and computation source. |
    | user_id | 2 | Uniform | 500000.50 | 500000.50 | Drop | Pure identifier with no predictive meaning. Must be removed from all models before training. |
    | age | 2 | Uniform | 49.00 | 49.00 | Medium | Near-uniform from 18–80 with spike at ~50. Fairly realistic distribution. Useful for demographic segmentation. |
    | income_level | 2 | Uniform | 104994.57 | 105013.00 | Medium | Flat spread from 25K–200K. Combine with monthly_spend to derive a saving/spending ratio for richer features. |
    | has_children | 2 | Binary | 0.40 | 0.00 | Medium | ~60% have no children. Important demographic feature that can influence purchasing patterns for family-oriented products. |
    | household_size | 2 | Uniform | 5.51 | 6.00 | Low | Flat 1–10 distribution. Unrealistic — large households should be rarer. Low informational value as-is. |
    | weekly_purchases | 2 | Uniform | 9.99 | 10.00 | Low | Perfectly uniform 0–20. Suspiciously flat. Needs validation before use; may not reflect real buying behavior. |
    | monthly_spend | 2 | Uniform | 2498.78 | 2498.00 | Medium | Flat 0–5000. Conceptually important but flat distribution limits discriminative power. Try interaction with income_level. |
    | cart_abandonment_rate | 2 | Skewed | 40.21 | 40.00 | High | Large spike at 0, then declining. Realistic pattern — many users never abandon. Strong behavioral feature for conversion models. |
    | review_writing_frequency | 2 | Skewed | 3.45 | 3.00 | Medium | Right-skewed, most users write 2–4 reviews. Reasonably realistic. Good proxy for user engagement depth. |
    | average_order_value | 2 | Uniform | 255.03 | 255.00 | Medium | Uniform 0–500. Conceptually valuable; combine with weekly_purchases to build a total revenue proxy. |
    | coupon_usage_frequency | 3 | Discrete | 2.00 | 2.00 | Medium | Only 5 discrete values (0–4) with equal frequency. Ordinal or one-hot encoding required. |
    | loyalty_program_member | 3 | Binary | 0.50 | 0.00 | High | Perfect 50/50 split. Strong segmentation feature — members vs non-members typically show significantly different behaviors. |
    | referral_count | 3 | Uniform | 5.00 | 5.00 | Low | Flat 0–10. Conceptually useful (active recommenders), but data needs real-world validation. |
    | weekend_shopper | 3 | Binary | 0.50 | 1.00 | Medium | 50/50 split. Temporal binary feature useful for weekend vs weekday promotion targeting. |
    | impulse_purchases_per_month | 3 | Skewed | 3.29 | 3.00 | Medium | Right-skewed, most users at 2–4x/month. Realistic enough to support buyer-type segmentation. |
    | browse_to_buy_ratio | 3 | Uniform | 55.00 | 55.00 | Low | Near-uniform with a slight spike at high values. Important concept but flat distribution reduces practical value. |
    | return_frequency | 3 | Uniform | 6.00 | 6.00 | Low | Flat 0–12. Unrealistic — high return frequency should be rarer. Low informational value in current form. |
    | brand_loyalty_score | 3 | Uniform | 5.00 | 5.00 | Low | Perfectly uniform 0–10. Constructed score with no natural distribution. Needs clear definition before use in models. |
    | impulse_buying_score | 3 | Discrete | 5.00 | 5.00 | Low | Mild U-shape pattern at 0 and 10. Slightly more realistic than other scores, but still clearly synthetic. |
    | environmental_consciousness | 3 | Uniform | 5.01 | 5.00 | Low | Uniform 0–10. Interesting concept for green consumer segmentation, but data is not informative as generated. |
    | health_conscious_shopping | 4 | Binary | 0.50 | 1.00 | Medium | 50/50 split. Useful lifestyle binary for segmenting health-oriented vs general product buyers. |
    | travel_frequency | 4 | Uniform | 5.99 | 6.00 | Low | Flat 0–12. Unrealistic. Could work as an active lifestyle proxy if validated with real data. |
    | hobby_count | 4 | Uniform | 2.50 | 2.00 | Low | Near-uniform 0–5. General demographic information; likely low predictive power for purchase behavior. |
    | social_media_influence_score | 4 | Uniform | 5.00 | 5.00 | Low | Perfectly flat 0–10. Synthetic score with zero discriminative value in current form. |
    | reading_habits | 4 | Uniform | 12.00 | 12.00 | Low | Uniform 0–24. Unrealistic — books per year should be right-skewed in a real population. |
    | exercise_frequency | 4 | Uniform | 3.50 | 4.00 | Low | Flat discrete values 0–7. Unrealistic — exercise frequency is typically right-skewed in general population. |
    | stress_from_financial_decisions | 4 | Uniform | 5.00 | 5.00 | Low | Uniform 0–10. Interesting concept but synthetic distribution removes predictive value. |
    | overall_stress_level | 4 | Uniform | 5.00 | 5.00 | Low | Near-uniform with minor spikes at 0 and 10. Slightly more realistic; could pair with financial stress feature. |
    | sleep_quality | 4 | Uniform | 6.50 | 6.00 | Low | Uniform range 4–9. Reasonable range but flat distribution. Could be combined with stress scores. |
    | physical_activity_level | 4 | Uniform | 4.99 | 5.00 | Low | Uniform 0–10. Same pattern as other lifestyle features — too flat to be informative. |
    | mental_health_score | 5 | Uniform | 5.00 | 5.00 | Low | Uniform 0–10. Ethically sensitive feature. Requires special consideration before use in targeting or segmentation. |
    | daily_session_time_minutes | 5 | Skewed | 60.01 | 60.00 | High | Near-uniform with large spike at ~120 min. The anomaly likely indicates a hard cap or systematic outlier — investigate before use. |
    | product_views_per_day | 5 | Uniform | 25.02 | 25.00 | Medium | Uniform with odd-even alternating pattern. Sampling artifact that needs examination. Conceptually useful for engagement modeling. |
    | ad_views_per_day | 5 | Uniform | 10.00 | 10.00 | Low | Flat 0–20. Combine with ad_clicks to derive CTR, which carries more signal than either column alone. |
    | ad_clicks_per_day | 5 | Discrete | 2.50 | 2.00 | Medium | Only 6 discrete values (0–5), uniformly distributed. Derive CTR ratio with ad_views for more meaningful signal. |
    | wishlist_items_count | 5 | Uniform | 10.00 | 10.00 | Low | Flat 0–20. Purchase intent proxy in concept, but flat distribution limits predictive power. |
    | cart_items_average | 5 | Uniform | 5.49 | 5.00 | Medium | Flat 1–10. Can proxy basket size. Combine with average_order_value for total basket value estimate. |
    | checkout_abandonments_per_month | 5 | Uniform | 5.00 | 5.00 | Low | Flat 0–10. Unrealistic — should be right-skewed. Low informational value in current form. |
    | purchase_conversion_rate | 5 | Uniform | 50.00 | 50.00 | Medium | Flat 0–100. Core e-commerce metric in concept, but flat distribution limits its direct usefulness as a feature. |
    | app_usage_frequency | 5 | Discrete | 3.50 | 3.00 | Medium | 7 discrete values (0–7), uniformly spread. Useful engagement proxy when combined with daily_session_time_minutes. |

    ---

    ## **Priority Summary**

    | Priority | Features |
    |---|---|
    | High | engagement_score, risk_score, premium_subscription, cart_abandonment_rate, loyalty_program_member, daily_session_time_minutes |
    | Medium | social_sharing_frequency, purchase_year, age, income_level, has_children, monthly_spend, review_writing_frequency, average_order_value, coupon_usage_frequency, weekend_shopper, impulse_purchases_per_month, health_conscious_shopping, product_views_per_day, ad_clicks_per_day, cart_items_average, purchase_conversion_rate, app_usage_frequency |
    | Low | notification_response_rate, account_age_months, return_rate, purchase_month, purchase_day, household_size, weekly_purchases, browse_to_buy_ratio, return_frequency, brand_loyalty_score, impulse_buying_score, environmental_consciousness, travel_frequency, hobby_count, social_media_influence_score, reading_habits, exercise_frequency, stress_from_financial_decisions, overall_stress_level, sleep_quality, physical_activity_level, mental_health_score, ad_views_per_day, wishlist_items_count, checkout_abandonments_per_month |
    | Drop | user_id |

    ---

    ## **Distribution Type Summary**

    | Distribution | Count | Features |
    |---|---|---|
    | Uniform | 26 | Most features — flat, synthetic-looking distributions |
    | Normal | 2 | engagement_score, risk_score |
    | Skewed | 4 | social_sharing_frequency, cart_abandonment_rate, review_writing_frequency, daily_session_time_minutes |
    | Binary | 8 | premium_subscription, purchase_year, has_children, loyalty_program_member, weekend_shopper, health_conscious_shopping |
    | Discrete | 4 | coupon_usage_frequency, impulse_buying_score, ad_clicks_per_day, app_usage_frequency |
    | Drop | 1 | user_id |

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Based on the descriptive statistics (mean, median, and distribution shape), the most appropriate business analysis approach is to focus on features that genuinely reflect user behavior rather than relying on overall averages.

    Most features in the dataset show a uniform (flat) distribution, meaning their values are evenly spread without any clear pattern. This makes them weak for business insights because they do not highlight meaningful differences between users. As a result, such features should not be used directly; they either need further transformation or should be excluded if they add no value.

    In contrast, features with more realistic distributions—such as normal (e.g., engagement_score, risk_score) and skewed distributions (e.g., cart_abandonment_rate, social_sharing_frequency)—are the primary sources of insight. These features reflect real-world behavior, where most users are relatively inactive and only a small portion are highly active. This enables effective customer segmentation, such as identifying high-value users, at-risk users, and passive users.

    Funnel analysis is also critical, particularly for understanding where users drop off in the purchasing process. A high cart abandonment rate suggests friction at the checkout stage or hesitation before completing a purchase. This can be addressed through strategies such as discounts, improved user experience, or retargeting campaigns.

    For revenue analysis, raw features like monthly_spend or average_order_value are not very informative due to their flat distribution. Instead, derived metrics should be created, such as estimated revenue (e.g., average order value multiplied by purchase frequency) or spending-to-income ratio. These provide a more realistic view of customer value.

    In terms of marketing effectiveness, metrics like ad views and ad clicks are not very useful on their own. However, when combined into ratios such as click-through rate (CTR), they become much more meaningful in evaluating campaign performance. Similarly, promotion-related features like coupon usage and impulse purchases can help identify customer types—for example, price-sensitive users versus impulse buyers.

    Overall, the key insight is that business decisions should not rely on raw data that appears structured but lacks meaningful patterns. The real value lies in transforming and combining features to uncover behavioral insights and enable effective segmentation. This approach leads to more accurate and actionable business decisions.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Demographic Analysis**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Age Distribution**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the 'age' column
    df['age'].describe()
    return


@app.cell
def _(df, plt, sns):
    # Plot the distribution of 'age' with mean and median
    sns.histplot(df['age'], bins=30, kde=True)

    plt.axvline(df['age'].mean(), linestyle='--', linewidth=1)
    plt.axvline(df['age'].median(), linestyle=':', linewidth=1)

    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The age distribution appears **almost perfectly uniform** across the range of 18 to 80 years, with a mean and median around 49–50 years. This indicates that each age group is represented in nearly equal proportions, with no visible concentration in younger or older segments.

    From a statistical perspective, this type of distribution is highly unusual in real-world data. Typically, age distributions tend to be skewed (for example, more users in younger or middle-age groups). A uniform pattern like this suggests that the data may be **synthetic or artificially balanced**, rather than naturally occurring. As a result, age in its current form provides **limited standalone predictive power**, because it does not highlight any dominant age segment or natural clustering.

    From a business perspective, this distribution implies that the dataset represents a **broad and evenly distributed customer base across all age groups**. In theory, this suggests that the product or service is designed for a wide audience, not targeting any specific age demographic. However, this also means that **age alone is not sufficient to drive business decisions**, since no particular age group stands out in terms of volume.

    The real value of age in this context comes when it is combined with other variables. For example, differences in spending behavior, engagement levels, or conversion rates across age groups may still exist, even if the population is evenly distributed. Segmenting users into age groups (such as young adults, middle-aged, and seniors) and analyzing their behavior can reveal actionable insights, such as which age group generates the highest revenue or shows the strongest engagement.

    Although the age distribution in this dataset is evenly spread and does not reflect a typical real-world pattern, it can still be used for business analysis with the right approach. Since no particular age group dominates, age should not be interpreted as a primary driver of customer behavior or used directly to draw conclusions about the target market.

    Instead, the data becomes more meaningful when age is used as a basis for segmentation rather than as an individual variable. Grouping users into broader age categories—such as young adults, mid-career individuals, and older users—allows the business to compare behaviors across segments. This approach helps uncover differences in engagement, spending patterns, or purchasing behavior that are not visible from the raw distribution alone.

    The value of age also increases significantly when combined with behavioral metrics. For example, analyzing how engagement, spending, or conversion rates vary across age groups can provide actionable insights, such as identifying which segments are more valuable or which require targeted marketing strategies. In this context, age acts as a supporting dimension that enhances the understanding of user behavior rather than serving as a standalone indicator.

    It is also important to focus on relative comparisons between age groups rather than absolute proportions. Since the dataset is artificially balanced, conclusions should be based on how different groups perform, not on how large they are. This ensures that the analysis remains relevant despite the limitations of the data.

    Overall, while the age distribution itself offers limited direct insight, it can still support meaningful business decisions when used for segmentation, combined with other variables, and interpreted with caution.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Gender Distribution**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the Gender Distribution
    df['gender'].value_counts(normalize=True)
    return


@app.cell
def _(df):
    # Chance the 'non-binary' value to 'other' for better visualization
    df['gender'] = df['gender'].replace('Non-binary', 'Other')
    df['gender'].value_counts(normalize=True)
    return


@app.cell
def _(df, plt, sns):
    # Visualize the gender distribution
    sns.set_style("whitegrid")

    ax_1 = sns.countplot(x='gender', data=df)

    # Add count and percentage annotations on top of the bars
    total = len(df)
    for p in ax_1.patches:
        count = int(p.get_height())
        pct = count / total * 100
        ax_1.annotate(f'{count}\n({pct:.1f}%)',
                    (p.get_x() + p.get_width() / 2, count),
                    ha='center', va='bottom', fontsize=9)

    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")

    plt.show()
    return (total,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The gender distribution shows a nearly perfect balance between male and female users, each representing approximately 48% of the total population, while the “Other” category accounts for about 4%. From a statistical perspective, this indicates a well-balanced dataset with no significant bias toward one of the primary gender groups. Such a distribution is beneficial for analysis and modeling, as it reduces the risk of skewed results driven by overrepresented categories. However, the relatively small proportion of the “Other” category suggests that insights for this group may be less reliable due to limited data.

    From a business standpoint, this balance implies that the company’s products or services appeal equally to both male and female customers, positioning the business within a broad, mass-market segment rather than a gender-specific niche. Despite this balance, it is important not to assume that both groups behave similarly. Differences in purchasing behavior, engagement levels, and conversion rates may still exist and should be explored further to uncover meaningful patterns.

    The presence of a smaller “Other” segment, while limited in size, still represents a notable group of users. This segment may offer opportunities for more inclusive marketing strategies or niche targeting, particularly if their behavior differs significantly from the larger groups.

    Overall, while the gender distribution itself does not immediately reveal actionable insights, it provides a strong foundation for deeper analysis. The real business value lies in examining how each gender group behaves across key metrics such as spending, engagement, and retention, rather than relying solely on the distribution proportions.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Country Distribution**

    ---
    """)
    return


@app.cell
def _(df, plt, sns):
    # Analyze the top 10 countries by customer count and their deviation from the average
    top_country = df['country'].value_counts().head(10)
    mean_val = top_country.mean()

    deviation = top_country - mean_val
    deviation = deviation.sort_values()

    plt.figure(figsize=(10, 6))
    ax_2 = sns.barplot(x=deviation.values, y=deviation.index)

    for i_1, v in enumerate(deviation.values):
        ax_2.text(v, i_1, f'{v:+.0f}', va='center')

    plt.title("Deviation from Average Customer Count")
    plt.xlabel("Difference from Mean")
    plt.ylabel("Country")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The country distribution shows how each market performs relative to the average customer count, revealing clear differences across regions. Some countries, such as Brazil, Canada, China, and Germany, are performing above the average, with Brazil standing out as the strongest contributor. This indicates that these markets have higher customer acquisition or stronger demand, making them key growth drivers for the business.

    On the other hand, several countries fall below the average, most notably Japan, which shows a significantly large negative deviation. This suggests a substantial gap in performance compared to other markets and may indicate underlying issues such as poor market fit, ineffective marketing strategies, or localization challenges. Other countries like the UK and India also underperform, though to a lesser extent, suggesting opportunities for optimization rather than critical concern.

    Meanwhile, countries such as the USA and France are positioned close to the average, indicating stable and consistent performance. These markets can serve as benchmarks for evaluating both high-performing and underperforming regions.

    From a business perspective, this distribution highlights that performance is not evenly spread across countries and that geographic factors play a significant role in customer acquisition. High-performing markets present opportunities for scaling and increased investment, while underperforming markets require deeper analysis to identify and address potential issues. Overall, the insight emphasizes the importance of tailoring strategies by region rather than applying a uniform global approach.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Income Level**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the income level distribution
    df['income_level'].value_counts(normalize=True)
    return


@app.cell
def _(df):
    # Create income groups based on the 'income_level' column
    df['income_level'].describe()
    return


@app.cell
def _(df, pd):
    # Create income groups based on the 'income_level' column
    bins = [10000, 40000, 80000, 120000, 160000, 200000]
    labels = ['Very Low', 'Low', 'Middle', 'Upper-Middle', 'High']

    df['income_group'] = pd.cut(df['income_level'], bins=bins, labels=labels)
    return (labels,)


@app.cell
def _(df, labels, plt, sns):
    # Visualize the income group distribution
    sns.set_style("whitegrid")

    ax_3 = sns.countplot(x='income_group', data=df, order=labels)

    total_1 = len(df)

    for p_1 in ax_3.patches:
        count_1 = int(p_1.get_height())
        pct_1 = count_1 / total_1 * 100
        ax_3.annotate(f'{count_1}\n({pct_1:.1f}%)',
                    (p_1.get_x() + p_1.get_width()/2, count_1),
                    ha='center', va='bottom', fontsize=9)

    plt.title("Customer Distribution by Income Level")
    plt.xlabel("Income Segment")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The income level distribution shows that customers are spread across five income segments—Very Low, Low, Middle, Upper-Middle, and High—using defined income bins. Most segments (Low to High) each account for roughly **21% of customers**, while the Very Low segment is noticeably smaller at around **15.8%**.

    From a statistical perspective, this distribution is **almost uniform across the main income groups**, with only a slight drop in the Very Low category. This suggests that the dataset has been **artificially balanced or evenly sampled**, rather than reflecting a naturally skewed income distribution typically seen in real populations (where lower-income groups are usually more dominant). As a result, income level in this dataset does not show strong natural variation and may have limited standalone explanatory power.

    From a business perspective, this distribution implies that the customer base is **evenly distributed across different purchasing power levels**, indicating a broad market reach. The relatively smaller Very Low segment suggests that the business may be less penetrated in the lowest income group, or that this segment is underrepresented in the data.

    However, the real value of income level lies not in the distribution itself, but in how different income groups behave. For example, higher-income segments are typically expected to contribute more to revenue through higher spending, while lower-income segments may be more price-sensitive and responsive to discounts or promotions. Therefore, income level should be used to **analyze differences in spending behavior, conversion rates, and product preferences across segments**.

    In summary, while the distribution appears clean and balanced, it is likely not fully representative of real-world conditions. Income level should not be used as a primary indicator on its own, but rather as a **segmentation variable** to support deeper behavioral and revenue analysis across different customer groups.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Spending Behavior Analysis**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the 'monthly_spend' column
    df['monthly_spend'].describe()
    return


@app.cell
def _(df, plt, sns):
    # Plot the distribution of 'monthly_spend' with mean and median
    sns.set_style("whitegrid")

    sns.histplot(df['monthly_spend'], bins=30, kde=True)

    plt.axvline(df['monthly_spend'].mean(), linestyle='--', linewidth=1)
    plt.axvline(df['monthly_spend'].median(), linestyle=':', linewidth=1)

    plt.title("Monthly Spend Distribution")
    plt.xlabel("Monthly Spend")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The monthly spending distribution shows a **nearly uniform pattern across the full range (0–5000)**, with an average around the midpoint (~2500). This means that customers are evenly spread across all spending levels, with no clear concentration of low, medium, or high spenders.

    From a statistical perspective, this distribution is **not realistic for real-world spending behavior**, where spending is typically right-skewed (most users spend low to moderate amounts, and only a few are high spenders). The flat distribution suggests that the data is likely **synthetic or artificially generated**, resulting in limited natural patterns. Because of this, monthly_spend as a standalone variable has **low discriminative power**, since it does not highlight dominant customer segments.

    From a business perspective, this means that the dataset does not immediately reveal which group of customers contributes the most revenue. In real scenarios, businesses usually rely on identifying high-value customers (top spenders), but in this case, such segmentation is not visible directly due to the uniform spread.

    To make this feature useful, monthly_spend should be transformed and analyzed in combination with other variables. For example, grouping customers into spending tiers (low, medium, high) can help create clearer segments. Additionally, combining monthly_spend with income_level can provide insight into purchasing power, such as identifying customers who spend a high proportion of their income. It can also be paired with behavioral features like engagement_score or purchase frequency to identify truly valuable customers.

    Overall, while the raw distribution of monthly_spend does not provide strong direct insights, it can still support meaningful business analysis when used as part of segmentation or combined with other variables.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Purchase Frequency (weekly_purchases)**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the 'weekly_purchases' column
    df['weekly_purchases'].describe()
    return


@app.cell
def _(df, plt, sns):
    # Plot the distribution of 'weekly_purchases' with mean and median
    sns.set_style("whitegrid")

    sns.histplot(df['weekly_purchases'], bins=21, discrete=True)

    plt.axvline(df['weekly_purchases'].mean(), linestyle='--', linewidth=1)
    plt.axvline(df['weekly_purchases'].median(), linestyle=':', linewidth=1)

    plt.title("Weekly Purchases Distribution")
    plt.xlabel("Weekly Purchases")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The weekly purchases distribution shows a **uniform spread across the range of 0 to 20 purchases per week**, with an average around 10. This means that customers are evenly distributed across all purchase frequencies, from very low to very high activity levels.

    From a statistical perspective, this pattern is **not typical of real-world behavior**. In reality, purchase frequency is usually right-skewed, where most customers make few purchases and only a small portion are highly active buyers. The flat distribution here suggests that the data is likely **synthetic or artificially balanced**, resulting in a lack of natural clustering. Because of this, weekly_purchases as a standalone feature has **limited ability to distinguish between customer segments**, since all activity levels are equally represented.

    From a business perspective, this distribution does not clearly identify core customer groups such as occasional buyers, regular buyers, or power users. Normally, businesses rely on such distinctions to drive strategies like retention campaigns or loyalty programs. However, in this case, those segments are not immediately visible from the raw data.

    To make this feature useful, it should be transformed into meaningful categories, such as low-frequency, medium-frequency, and high-frequency buyers. Additionally, combining weekly_purchases with other variables can generate stronger insights. For example, pairing it with average_order_value can help estimate customer value, while combining it with engagement_score can distinguish between highly active and less engaged users.

    Overall, while the raw distribution of weekly_purchases provides limited direct insight due to its uniform nature, it can still support business analysis when used for segmentation or combined with other behavioral and financial metrics.


    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Marketing Behavior Analysis**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Coupon Usage**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the 'coupon_usage_frequency' column
    df['coupon_usage_frequency'].describe()
    return


@app.cell
def _(df, plt, sns, total):
    # Plot the distribution of 'coupon_usage_frequency' with mean and median
    sns.set_style("whitegrid")

    ax_4 = sns.histplot(
        df['coupon_usage_frequency'],
        bins=range(int(df['coupon_usage_frequency'].min()), int(df['coupon_usage_frequency'].max()) + 2),
        discrete=True
    )

    # Add count and percentage annotations on top of the bars
    for p_2 in ax_4.patches:
        count_2 = int(p_2.get_height())
        if count_2 > 0:
            pct_2 = count_2 / total * 100
            ax_4.text(p_2.get_x() + p_2.get_width()/2, count_2,
                    f'{count_2}\n({pct_2:.1f}%)',
                    ha='center', va='bottom', fontsize=8)

    plt.title("Coupon Usage Frequency Distribution")
    plt.xlabel("Coupon Usage Frequency")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The coupon usage distribution shows that customers are evenly spread across all usage levels (0 to 4), with each category representing 20% of the population. This indicates a perfectly balanced distribution where no single group—whether non-users or frequent coupon users—dominates.

    From a statistical perspective, this uniform pattern is highly unlikely in real-world scenarios. Typically, coupon usage is skewed, with most customers either rarely using coupons or using them occasionally, and only a smaller portion using them frequently. The equal distribution across all categories suggests that the data is likely synthetic or artificially structured, which limits its ability to reveal natural behavioral patterns.

    From a business perspective, this distribution does not immediately indicate whether customers are generally price-sensitive or not, since all levels of coupon usage are equally represented. As a result, it is not possible to directly identify dominant customer groups such as “discount-driven users” or “full-price buyers” from this data alone.

    However, the feature can still be valuable when used for segmentation. Customers can be grouped into categories such as non-users (0), occasional users (1–2), and frequent users (3–4). This allows the business to differentiate between price-sensitive customers and those less influenced by promotions. When combined with other variables—such as monthly spending or purchase frequency—coupon usage can help identify patterns like high-spending users who rely heavily on discounts or loyal customers who purchase without incentives.

    In summary, while the raw distribution of coupon usage does not provide strong standalone insights due to its uniform nature, it remains useful as a segmentation variable. Its real business value emerges when it is combined with behavioral and financial metrics to better understand customer sensitivity to pricing and promotions.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Ads & Response**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the 'ad_views_per_day' and 'ad_clicks_per_day' columns
    df[['ad_views_per_day', 'ad_clicks_per_day']].describe()
    return


@app.cell
def _(df, plt):
    # Visualize the relationship between ad views and ad clicks
    plt.figure(figsize=(9, 6))

    plt.hexbin(
        df['ad_views_per_day'],
        df['ad_clicks_per_day'],
        gridsize=40,
        mincnt=1
    )

    plt.colorbar(label='Number of Customers')
    plt.title("Ad Views vs Ad Clicks per Day")
    plt.xlabel("Ad Views per Day")
    plt.ylabel("Ad Clicks per Day")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The relationship between ad views and ad clicks per day shows a **discrete and evenly distributed pattern**, where ad clicks range from 0 to 5 and ad views from 0 to 20, with customer counts relatively similar across all combinations. There is no clear upward trend or concentration that would indicate a strong relationship between the number of ads viewed and the number of clicks.

    From a statistical perspective, this suggests that the variables are **weakly correlated or potentially independent** in this dataset. In real-world scenarios, we would typically expect a positive relationship—more ad views leading to more clicks—but this pattern is not clearly visible here. The evenly spread distribution and similar customer counts across combinations indicate that the data may be **synthetic or artificially structured**, limiting its ability to capture realistic user behavior.

    From a business perspective, this means that **ad exposure alone does not appear to drive user engagement (clicks)** in this dataset. Simply increasing the number of ads shown to users may not lead to higher interaction, which challenges the assumption that more impressions automatically result in better performance.

    To extract meaningful insight, the focus should shift from raw counts to derived metrics, particularly the **click-through rate (CTR)**, calculated as clicks divided by views. CTR provides a clearer measure of ad effectiveness and user responsiveness. Additionally, combining this data with other features—such as engagement level, user segments, or purchase behavior—can help identify which types of users are more responsive to ads.

    In summary, while the raw relationship between ad views and clicks does not provide strong direct insight due to its flat and synthetic nature, it still highlights an important business implication: **ad quantity alone is not sufficient to drive engagement**. More value can be obtained by focusing on ad quality, targeting, and user segmentation rather than simply increasing exposure.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Risk Behavior Analysis**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Cart Abandonment**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the 'cart_abandonment_rate' column
    df['cart_abandonment_rate'].describe()
    return


@app.cell
def _(df, plt, sns):
    # Plot the distribution of 'cart_abandonment_rate' with mean and median
    sns.set_style("whitegrid")

    sns.histplot(df['cart_abandonment_rate'], bins=30, kde=True)

    mean_val_2 = df['cart_abandonment_rate'].mean()
    median_val_2 = df['cart_abandonment_rate'].median()

    plt.axvline(mean_val_2, linestyle='--', linewidth=1, label=f"Mean: {mean_val_2:.2f}")
    plt.axvline(median_val_2, linestyle=':', linewidth=1, label=f"Median: {median_val_2:.2f}")

    plt.title("Cart Abandonment Rate Distribution")
    plt.xlabel("Cart Abandonment Rate")
    plt.ylabel("Frequency")

    plt.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The cart abandonment rate distribution shows a mixed pattern, with a noticeable spike at very low values (near 0%) and a relatively even spread across higher ranges up to around 90%. The mean (~40.21) and median (~40.00) are almost identical, indicating that overall the distribution is fairly balanced, but the spike at low values suggests the presence of a distinct group of users who rarely abandon their carts.

    From a statistical perspective, this distribution is partially realistic compared to other variables. The concentration at low abandonment rates reflects a segment of highly committed users who consistently complete purchases. At the same time, the wide spread across mid-to-high values indicates variability in user behavior, where many users frequently abandon carts. This combination creates a more informative feature, as it introduces natural variation and segmentation potential.

    From a business perspective, this is a highly valuable behavioral metric because it directly reflects friction in the purchase process. The group with near-zero abandonment likely represents loyal or highly motivated customers, while users with high abandonment rates may be facing barriers such as pricing concerns, complicated checkout processes, lack of trust, or insufficient incentives.

    This distribution enables clear segmentation of customers into meaningful groups: low abandonment users (high intent), moderate abandonment users (undecided or price-sensitive), and high abandonment users (at risk of dropping off). Each group can be targeted with different strategies. For example, high abandonment users may benefit from retargeting campaigns, discounts, or checkout simplification, while low abandonment users can be targeted for upselling or loyalty programs.

    Overall, unlike many other features in the dataset, cart abandonment rate provides strong and actionable business insight. It highlights both opportunities for conversion optimization and areas where the user experience may need improvement, making it a critical metric for driving revenue growth.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Return Rate**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the 'return_rate' column
    df['return_rate'].describe()
    return


@app.cell
def _(df, plt, sns):
    # Plot the distribution of 'return_rate' with mean and median
    sns.set_style("whitegrid")

    data_rr = df['return_rate'] * 100  # Change to percentage for better readability

    mean_val_3 = data_rr.mean()
    median_val_3 = data_rr.median()

    sns.histplot(data_rr, bins=30, kde=True)

    plt.axvline(mean_val_3, linestyle='--', linewidth=1, label=f"Mean: {mean_val_3:.1f}%")
    plt.axvline(median_val_3, linestyle=':', linewidth=1, label=f"Median: {median_val_3:.1f}%")

    plt.title("Return Rate Distribution")
    plt.xlabel("Return Rate (%)")
    plt.ylabel("Frequency")

    plt.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The return rate distribution shows a **perfectly uniform spread across a very wide range (0% to 10,000%)**, with the mean and median both around 5000%. This indicates that all values are equally represented, with no clustering or concentration in any specific range.

    From a statistical perspective, this distribution is **highly unrealistic and problematic**. In real-world scenarios, return rates are typically low (often below 20–30%) and heavily right-skewed, with most customers returning few or no items. A range extending to 10,000% is not practically meaningful and strongly suggests that the data is **synthetic, mis-scaled, or incorrectly calculated**. The fact that the mean and median are identical further reinforces that this is an artificially uniform dataset with no natural variation or behavioral pattern.

    From a business perspective, this feature in its current form is **not usable for decision-making**. It does not provide any meaningful insight into customer behavior, product quality, or operational issues. In reality, return rate is a critical metric used to identify problems such as defective products, mismatched expectations, or poor customer satisfaction. However, with this kind of distribution, it is impossible to distinguish between normal and problematic behavior.

    The primary implication is that this variable requires **immediate validation and correction** before it can be used. Possible issues include incorrect scaling (e.g., percentages multiplied incorrectly), data generation errors, or placeholder values. Once corrected, return rate could become a highly valuable feature for identifying high-risk customers, problematic products, or inefficiencies in the fulfillment process.

    In summary, while return rate is conceptually an important business metric, the current distribution renders it **statistically invalid and business-wise unusable**. It should be excluded from analysis until the data quality issue is resolved.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Loyalty & Premium User**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Loyalty Program**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the 'loyalty_program_member' column
    df['loyalty_program_member'].value_counts(normalize=True)
    return


@app.cell
def _(df, plt, sns):
    # Visualize the loyalty program membership distribution
    sns.set_style("whitegrid")

    ax_5 = sns.countplot(
        x='loyalty_program_member',
        data=df,
        order=df['loyalty_program_member'].value_counts().index
    )

    total_2 = len(df)

    # label count + persen
    for p_3 in ax_5.patches:
        count_3 = int(p_3.get_height())
        pct_4 = count_3 / total_2 * 100
        ax_5.annotate(f'{count_3}\n({pct_4:.1f}%)',
                    (p_3.get_x() + p_3.get_width()/2, count_3),
                    ha='center', va='bottom', fontsize=9)

    plt.title("Loyalty Program Membership Distribution")
    plt.xlabel("Loyalty Program Member")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The loyalty program membership distribution shows an almost perfectly equal split between non-members (50.0%) and members (50.0%). This indicates that half of the customer base is enrolled in the loyalty program, while the other half is not.

    From a statistical perspective, this is a **balanced binary distribution**, which is generally favorable for analysis and modeling because there is no class imbalance. However, such a perfectly even split is somewhat **unusual in real-world scenarios**, where loyalty program participation is often either lower (early-stage programs) or skewed toward active users (mature programs). This suggests the data may be **artificially balanced**, similar to other variables in the dataset.

    From a business perspective, this distribution provides a strong foundation for **comparative analysis between members and non-members**. It allows the business to directly evaluate the impact of the loyalty program on key metrics such as spending, purchase frequency, retention, and engagement. For example, if members show higher spending or lower cart abandonment, this would validate the effectiveness of the program.

    The equal split also highlights a clear opportunity: since half of the customers are not enrolled, there is significant room to **expand membership and increase customer lifetime value**. Non-members represent a target segment for conversion campaigns, such as incentives for joining, exclusive benefits, or personalized offers.

    Overall, while the distribution itself does not indicate dominance of one group, it is highly useful from a business standpoint. The real insight lies in comparing behaviors between members and non-members, which can reveal whether the loyalty program is driving meaningful value and where it can be optimized or expanded.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Premium Subscription**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the 'premium_subscription' column
    df['premium_subscription'].value_counts(normalize=True)
    return


@app.cell
def _(df, plt, sns):
    # Visualize the premium subscription distribution
    sns.set_style("whitegrid")

    ax_6 = sns.countplot(
        x='premium_subscription',
        data=df,
        order=df['premium_subscription'].value_counts().index
    )

    total_3 = len(df)

    # label count + persentase
    for p_4 in ax_6.patches:
        count_4 = int(p_4.get_height())
        pct_5 = count_4 / total_3 * 100
        ax_6.annotate(f'{count_4}\n({pct_5:.1f}%)',
                    (p_4.get_x() + p_4.get_width()/2, count_4),
                    ha='center', va='bottom', fontsize=9)

    plt.title("Premium Subscription Distribution")
    plt.xlabel("Premium Subscription")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The premium subscription distribution shows that approximately **64.1% of customers are non-premium users**, while **35.9% are premium subscribers**. This indicates a clear imbalance, with the majority of the customer base not enrolled in the premium offering.

    From a statistical perspective, this is a **moderately imbalanced binary distribution**, which is more realistic compared to other features in the dataset. It reflects a typical adoption pattern where only a portion of users opt into paid or premium services. This type of distribution is informative because it creates a natural distinction between two groups with potentially different behaviors.

    From a business perspective, this distribution provides several important insights. First, the relatively high proportion of non-premium users suggests a **large opportunity for conversion and revenue growth**. Since over half of the users are not subscribed, targeted strategies such as free trials, feature previews, or personalized offers could be used to encourage upgrades.

    Second, premium users likely represent a **higher-value segment**, potentially contributing more in terms of spending, engagement, or retention. This makes it important to analyze how premium users differ from non-premium users across key metrics. If premium users show significantly better performance, it reinforces the value of the subscription model and supports further investment in premium features.

    At the same time, the fact that premium adoption is already at around 36% indicates that the offering has **established traction** and is not in an early-stage adoption phase. This suggests that the product-market fit for the premium service is reasonably strong, but still has room for expansion.

    Overall, this distribution highlights both **a solid existing premium user base and a significant untapped segment**. The key business opportunity lies in understanding what drives users to convert and leveraging that insight to increase premium adoption while maintaining value for existing subscribers.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Loyalty vs Spend**

    ---
    """)
    return


@app.cell
def _(df, plt, sns):
    # Analyze the relationship between loyalty program membership and monthly spend
    sns.set_style("whitegrid")

    ax_7 = sns.boxplot(
        x='loyalty_program_member',
        y='monthly_spend',
        data=df
    )

    # Add mean points to the boxplot
    sns.pointplot(
        x='loyalty_program_member',
        y='monthly_spend',
        data=df,
        estimator='mean',
        color='red',
        markers='D',
        linestyles=''
    )

    plt.title("Monthly Spend by Loyalty Program Membership")
    plt.xlabel("Loyalty Program Member")
    plt.ylabel("Monthly Spend")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The comparison of monthly spend between loyalty program members and non-members shows that **both groups have nearly identical spending patterns**. The median and mean values are almost the same, and the spread of data (from low to high spenders) is also very similar for both groups.

    From a statistical perspective, this indicates that **loyalty membership does not have a strong impact on spending behavior in this dataset**. There is no visible shift in central tendency (mean/median) or distribution between the two groups. In other words, being a member or not does not significantly differentiate how much customers spend per month. This suggests a **weak or negligible correlation** between loyalty membership and spending.

    From a business perspective, this is a critical insight. Ideally, a loyalty program is expected to increase customer value—either by encouraging higher spending, more frequent purchases, or stronger retention. However, in this case, the program does not appear to be delivering additional monetary value, at least in terms of monthly spend.

    This could imply several possibilities. The loyalty program may not be attractive enough, its benefits may not be clearly perceived by users, or it may be focused on aspects other than spending (such as engagement or retention). It is also possible that the dataset itself, which contains several uniform distributions, limits the visibility of real behavioral differences.

    The key takeaway is that the loyalty program, in its current state, **does not appear to drive increased spending**, which raises questions about its effectiveness. From a business standpoint, this suggests a need to re-evaluate the program—such as enhancing incentives, introducing tiered rewards, or personalizing benefits—to ensure it meaningfully influences customer behavior.

    Overall, while the comparison does not show a positive impact on spending, it provides a valuable direction for improvement and highlights an opportunity to strengthen the loyalty strategy.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Data Remediation**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Data Quality**

    ---
    """)
    return


@app.cell
def _(df):
    # Analyze the relationship between premium subscription and weekly purchases
    numeric_cols = [
        'age', 'weekly_purchases', 'monthly_spend', 'cart_abandonment_rate',
        'average_order_value', 'coupon_usage_frequency', 'referral_count',
        'impulse_purchases_per_month', 'browse_to_buy_ratio', 'brand_loyalty_score',
        'impulse_buying_score', 'social_media_influence_score',
        'overall_stress_level', 'mental_health_score', 'daily_session_time_minutes',
        'product_views_per_day', 'ad_views_per_day', 'ad_clicks_per_day',
        'wishlist_items_count', 'cart_items_average',
        'checkout_abandonments_per_month', 'purchase_conversion_rate',
        'notification_response_rate', 'account_age_months', 'social_sharing_frequency',
        'return_rate', 'engagement_score', 'risk_score', 'recency_days'
    ]

    numeric_cols = [col for col in numeric_cols if col in df.columns]

    # Descriptive statistics with skewness and kurtosis
    desc_stats = df[numeric_cols].describe().T
    desc_stats['skewness'] = df[numeric_cols].skew()
    desc_stats['kurtosis'] = df[numeric_cols].kurtosis()
    desc_stats
    return (numeric_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Based on the overall descriptive statistics (including mean, skewness, and kurtosis), it is clear that not all variables provide equal value for business analysis. Most variables exhibit very flat (uniform) distributions, which means they lack natural patterns and have limited ability to differentiate user behavior. Therefore, it is essential to identify the most informative variables and apply the right approach to make the data useful.

    The most important variables in this dataset are those with more realistic distributions and stronger behavioral signals. In particular, **engagement_score** and **risk_score** stand out as the most valuable features. Their distributions are closer to normal compared to other variables, suggesting that they contain richer information. These variables are likely derived from multiple user activities, making them highly relevant for prediction, segmentation, and overall customer evaluation.

    In addition, variables such as **cart_abandonment_rate**, **impulse_purchases_per_month**, and **social_sharing_frequency** are also important because they capture meaningful variations in user behavior. These features help explain how users interact with the platform—for example, whether they tend to complete purchases, make impulsive decisions, or engage socially. From a business perspective, these variables are especially useful for optimizing the conversion funnel and designing retargeting strategies.

    On the other hand, variables like **monthly_spend**, **weekly_purchases**, and **average_order_value** are conceptually critical because they relate directly to revenue. However, due to their uniform distributions, they do not provide strong insights when used individually. To make them useful, they should be transformed into derived metrics, such as estimated revenue (by combining purchase frequency and order value) or spending-to-income ratios. This approach helps uncover patterns that are not visible in the raw data.

    Demographic variables such as **age** and **income_level** also lack strong distribution patterns and should not be used as primary drivers of business decisions. However, they can still serve as supporting variables in segmentation, for example when comparing behavior across age groups or income levels.

    To address the limitations of flat distributions, the key solution is to apply **feature engineering and segmentation**. This includes grouping continuous variables into meaningful categories (e.g., low, medium, high spenders) and combining multiple features to create more informative metrics. By doing so, hidden patterns can emerge, enabling more actionable insights.

    In conclusion, the business value of this dataset does not lie in individual variables, but in how they are transformed, combined, and analyzed. By focusing on strong behavioral features and applying proper feature engineering, the dataset can still support meaningful and actionable business decisions.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Shapiro-Wilk Normality Test**

    ---
    """)
    return


@app.cell
def _(df, numeric_cols, pd, shapiro):
    # Shapiro-Wilk normality test
    normality_results = []

    for num_col in numeric_cols:
        stat, p_value = shapiro(df[num_col].dropna())
        normality_results.append({
            'column': num_col,
            'shapiro_stat': stat,
            'p_value': p_value,
            'is_normal_approx': p_value > 0.05
        })

    # Create a DataFrame to display the normality test results
    normality_df = pd.DataFrame(normality_results).sort_values('p_value')
    normality_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The results of the Shapiro-Wilk normality test indicate that all variables in the dataset are statistically non-normal, as shown by the p-values equal to zero across all features. This means that each variable significantly deviates from a normal distribution. Although some variables, such as **engagement_score** and **risk_score**, have Shapiro statistics very close to 1, they are still classified as non-normal due to the extremely large sample size, which makes the test highly sensitive to even minor deviations.

    From a statistical perspective, this confirms that the data does not follow a natural bell-shaped (normal) distribution. Instead, most variables exhibit patterns that are either flat (uniform) or discrete in nature. This aligns with earlier findings from skewness and kurtosis, where values were close to zero for skewness and negative for kurtosis, indicating symmetrical but flat distributions. In other words, the data lacks strong concentration around specific values and does not show natural clustering.

    From an analytical standpoint, this has important implications. Traditional statistical methods that assume normality, such as t-tests or ANOVA, may not be appropriate without additional validation or transformation. Instead, non-parametric methods or machine learning approaches that do not rely on distributional assumptions are more suitable. Additionally, since the distributions are flat, relying on averages alone can be misleading. Measures such as the median or percentile-based segmentation provide a more reliable understanding of the data.

    From a business perspective, the lack of normality suggests that the dataset does not naturally highlight dominant customer behaviors or segments. This means that insights cannot be derived directly from raw distributions. Instead, value must be extracted through segmentation, comparison between groups, and the creation of derived metrics. Variables like **engagement_score** and **risk_score**, despite not being perfectly normal, remain the most informative because they are closer to natural distributions and likely capture aggregated behavioral signals.

    Overall, while the absence of normality might seem like a limitation, it does not prevent meaningful analysis. Rather, it requires a shift in approach—from relying on standard statistical assumptions to using more robust, flexible methods that focus on relationships, segmentation, and feature interactions.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Outlier Detection**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Z-score Method**

    ---
    """)
    return


@app.cell
def _(df, np, numeric_cols, pd, stats):
    # Outlier detection using Z-score method
    outlier_summary_z = []

    for num_col_1 in numeric_cols:
        series = df[num_col_1].dropna()
        if series.std() == 0:
            outlier_count = 0
        else:
            z_scores = np.abs(stats.zscore(series))
            outlier_count = (z_scores > 3).sum()

        outlier_summary_z.append({
            'column': num_col_1,
            'outlier_count_zscore': outlier_count,
            'outlier_pct_zscore': outlier_count / len(series) * 100 if len(series) > 0 else 0
        })

    outlier_z_df = pd.DataFrame(outlier_summary_z).sort_values('outlier_pct_zscore', ascending=False)
    outlier_z_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **IQR Method**

    ---
    """)
    return


@app.cell
def _(df, numeric_cols, pd):
    # Outlier detection using IQR method
    outlier_summary_iqr = []

    for num_col_2 in numeric_cols:
        series_IQR = df[num_col_2].dropna()
        Q1 = series_IQR.quantile(0.25)
        Q3 = series_IQR.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (series_IQR < lower_bound) | (series_IQR > upper_bound)
        outlier_count_1 = outlier_mask.sum()

        outlier_summary_iqr.append({
            'column': num_col_2,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count_iqr': outlier_count_1,
            'outlier_pct_iqr': outlier_count_1 / len(series_IQR) * 100 if len(series_IQR) > 0 else 0
        })

    outlier_iqr_df = pd.DataFrame(outlier_summary_iqr).sort_values('outlier_pct_iqr', ascending=False)
    outlier_iqr_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    `engagement_score` and `risk_score` are the only ones that have outliers. In this dataset, user differences do not come from extreme values, but must be created through segmentation and feature combinations.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Feature Engineering 1**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Click-through Proxy**

    ---
    """)
    return


@app.cell
def _(df):
    # Create a new column for click-through rate (CTR)
    df['ctr'] = df['ad_clicks_per_day'] / (df['ad_views_per_day'] + 1)
    df['ctr']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    `Click-through Rate (CTR)` is used to see the effectiveness of exposure to clicks.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Revenue proxy**

    ---
    """)
    return


@app.cell
def _(df):
    # Create a new column for revenue proxy by multiplying monthly spend with purchase conversion rate
    df['revenue_proxy'] = df['monthly_spend'] * df['purchase_conversion_rate']
    df['revenue_proxy']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    `revenue_proxy` combines spending and conversion opportunities.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Spend per Order Tendency**

    ---
    """)
    return


@app.cell
def _(df):
    # Spend per order tendency
    df['spend_to_aov_ratio'] = df['monthly_spend'] / (df['average_order_value'] + 1)
    df['spend_to_aov_ratio']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    `spend_to_aov_ratio` helps to see how much monthly spend is relative to the average order value.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Customer Value Score**

    ---
    """)
    return


@app.cell
def _(df):
    # Value score combining purchase frequency, average order value, and conversion rate
    df['value_score'] = (
        df['weekly_purchases'] *
        df['average_order_value'] *
        df['purchase_conversion_rate']
    )
    df['value_score']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This feature measures customer value more comprehensively because it not only looks at spending, but also:

    - How often they purchase

    - How large the transaction value is

    - How effectively those activities translate into purchases

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Engagement Intensity**

    ---
    """)
    return


@app.cell
def _(df):
    # Engagement intensity combining engagement score and daily session time
    df['engagement_intensity'] = (
        df['engagement_score'] *
        df['daily_session_time_minutes']
    )
    df['engagement_intensity']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    While `engagement_score` indicates the quality of an interaction, `engagement_intensity` adds a dimension of duration. Users who are engaged and spend a long time are generally more valuable than users who are only active briefly.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Activity Score**

    ---
    """)
    return


@app.cell
def _(df):
    # Activity score combining product views, ad views, and app usage frequency
    df['activity_score'] = (
        df['product_views_per_day'] +
        df['ad_views_per_day'] +
        df['app_usage_frequency']
    )
    df['activity_score']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This feature is used to view the intensity of user activity on the platform in general. It's very useful for segmentation and behavioral analysis.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Abandonment Impact**

    ---
    """)
    return


@app.cell
def _(df):
    # Abandonment impact combining cart abandonment rate and checkout abandonments per month
    df['abandonment_impact'] = (
        df['cart_abandonment_rate'] *
        df['checkout_abandonments_per_month']
    )
    df['abandonment_impact']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    If users have a high abandonment rate and a high number of abandonments, then the bottleneck in the funnel is much more serious than just looking at one or the other.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Conversion Efficiency**

    ---
    """)
    return


@app.cell
def _(df):
    # Conversion efficiency combining weekly purchases and product views
    df['conversion_efficiency'] = (
        df['weekly_purchases'] /
        (df['product_views_per_day'] + 1)
    )
    df['conversion_efficiency']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    This feature measures how effectively browsing activity converts into purchases. It's more operational than simply using the `purchase_conversion_rate`.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Impulse Ratio**

    ---
    """)
    return


@app.cell
def _(df):
    # Impulse purchase ratio combining impulse purchases and weekly purchases
    df['impulse_ratio'] = (
        df['impulse_purchases_per_month'] /
        (df['weekly_purchases'] * 4 + 1)
    )
    df['impulse_ratio']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Since `weekly_purchases` is weekly, multiply it by 4 as a simple approximation for estimating monthly purchases. A high ratio indicates that most transactions are impulsive.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Discount Sensitivity**

    ---
    """)
    return


@app.cell
def _(df):
    # Discount sensitivity combining coupon usage frequency and weekly purchases
    df['discount_sensitivity'] = (
        df['coupon_usage_frequency'] /
        (df['weekly_purchases'] + 1)
    )
    df['discount_sensitivity']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    If a user frequently uses coupons but their total purchases are low, they're likely very price-sensitive. This is useful for segmenting promo vs. premium buyers.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Risk-Adjusted Value**

    ---
    """)
    return


@app.cell
def _(df):
    # Risk-adjusted value combining revenue proxy and risk score
    df['risk_adjusted_value'] = (
        df['revenue_proxy'] *
        (1 - df['risk_score'])
    )
    df['risk_adjusted_value']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Users with high revenues aren't necessarily ideal if they're also high risk. This feature helps prioritize valuable and relatively safe customers.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Churn Proxy**

    ---
    """)
    return


@app.cell
def _(df, np):
    # Churn proxy combining cart abandonment rate and engagement score
    df['churn_proxy'] = np.where(
        (df['cart_abandonment_rate'] > df['cart_abandonment_rate'].median()) &
        (df['engagement_score'] < df['engagement_score'].median()),
        1, 0
    )
    df['churn_proxy']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    High abandonment = high friction
    Low engagement = low interest

    The combination of the two is an early signal of churn risk.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Correlation & Transformation**

    ---
    """)
    return


@app.cell
def _(df):
    # Define the list of feature columns for further analysis or modeling
    feature_cols_FE = [
        'ctr', 'revenue_proxy', 'spend_to_aov_ratio', 'value_score',
        'engagement_intensity', 'activity_score', 'abandonment_impact',
        'conversion_efficiency', 'impulse_ratio', 'discount_sensitivity',
        'risk_adjusted_value', 'churn_proxy'
    ]

    feature_cols_FE = [col for col in feature_cols_FE if col in df.columns]
    return (feature_cols_FE,)


@app.cell
def _(df, np):
    # Apply log transformation to skewed features to reduce the impact of outliers and improve normality
    for day_col in ['monthly_spend', 'weekly_purchases', 'ad_views_per_day', 'ad_clicks_per_day']:
        if day_col in df.columns:
            df[f'log_{day_col}'] = np.log1p(df[day_col])
    return


@app.cell
def _(df, feature_cols_FE):
    # Calculate Pearson and Spearman correlation
    pearson_corr = df[feature_cols_FE + ['monthly_spend']].corr(method='pearson')['monthly_spend'].sort_values(ascending=False)
    spearman_corr = df[feature_cols_FE + ['monthly_spend']].corr(method='spearman')['monthly_spend'].sort_values(ascending=False)
    return pearson_corr, spearman_corr


@app.cell
def _(pearson_corr):
    # Show pearson_corr 
    print("Pearson correlation with monthly_spend")
    pearson_corr
    return


@app.cell
def _(spearman_corr):
    # Show pearson_corr 
    ("Spearman correlation with monthly_spend")
    spearman_corr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The Spearman correlation results provide a deeper understanding of the relationship between the engineered features and **monthly_spend**, especially in cases where relationships are not linear. Unlike Pearson correlation, which measures linear relationships, Spearman focuses on rank-based (monotonic) relationships. This makes it more suitable for datasets like this, where many variables are flat or not normally distributed.

    From the results, it is evident that only a small number of features show meaningful relationships with monthly spending. The strongest relationship is observed in **spend_to_aov_ratio**, which has a high correlation of approximately 0.69. This indicates that as the ratio increases, monthly spending also tends to increase in a consistent ranking pattern, even if the relationship is not strictly linear. This feature emerges as one of the most informative indicators of customer spending behavior.

    Similarly, **revenue_proxy** also shows a strong correlation (around 0.66), which is expected since it is directly derived from monthly spending. While this confirms that the feature engineering is structurally correct, it also suggests redundancy, meaning it does not provide new independent insight but rather reinforces the original variable.

    Another important feature is **risk_adjusted_value**, which shows a moderately strong correlation (around 0.58). This suggests that customers who have higher value and lower risk tend to spend more. From a business perspective, this feature is particularly useful for identifying high-quality customers who are both valuable and less risky.

    On the other hand, most of the remaining features—such as **discount_sensitivity**, **churn_proxy**, **conversion_efficiency**, **CTR**, **activity_score**, **abandonment_impact**, **impulse_ratio**, and **engagement_intensity**—show correlations very close to zero. This indicates that these variables do not have a meaningful relationship with monthly spending, even when considering non-linear patterns. In other words, variations in these features do not correspond to changes in spending behavior.

    An important takeaway from this analysis is that while Pearson correlation suggested almost no relationships, Spearman correlation reveals that some meaningful patterns do exist, but they are limited and primarily based on ranking rather than direct proportional changes. This confirms that the dataset contains weak structural relationships overall, with only a few features carrying significant signal.

    From a business perspective, this means that monthly spending in this dataset is influenced by only a narrow set of factors, and many commonly assumed drivers—such as engagement, activity, or marketing interactions—do not appear to play a role. This is not typical of real-world scenarios and suggests that the dataset may lack realistic behavioral dependencies.

    In conclusion, the findings highlight that only a few engineered features, particularly those directly tied to spending structure or adjusted value, are meaningful. Most other features do not contribute to explaining spending behavior. This reinforces the importance of careful feature selection and suggests that further improvements should focus on creating more behaviorally connected features or validating the underlying data structure.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Feature Engineering 2**

    ---
    """)
    return


@app.cell
def _(np, pd):
    # Define a function to safely perform division and handle division by zero cases
    def safe_divide(numerator, denominator):
        denominator = denominator.replace(0, np.nan) if isinstance(denominator, pd.Series) else denominator
        return numerator / denominator

    return (safe_divide,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The safe_divide function was created to avoid errors or invalid results when dividing by zero in data analysis.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Value & Revenue Feature**

    ---
    """)
    return


@app.cell
def _(df, safe_divide):
    # True Revenue
    df['true_revenue'] = df['weekly_purchases'] * df['average_order_value']

    # Adjusted Revenue
    df['adjusted_revenue'] = df['true_revenue'] * df['purchase_conversion_rate']

    # Revenue Efficiency
    df['revenue_efficiency'] = safe_divide(df['monthly_spend'], df['true_revenue'])

    # Revenue per Session
    df['revenue_per_session'] = safe_divide(df['monthly_spend'], df['daily_session_time_minutes'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. True Revenue

        Measures the estimated actual revenue based on purchase frequency and average transaction value (AOV). In theory, this metric reflects the customer's monetary value in models such as RFM (Recency, Frequency, Monetary).

    2. Adjusted Revenue

        Adjusts revenue by conversion probability to reflect a more realistic value. This concept originates from expected value in probabilistic modeling, where a value is multiplied by the probability of occurrence.

    3. Revenue Efficiency

        Measures how efficiently spending generates revenue. In theory, this is related to the concept of efficiency ratio, which is often used in profitability analysis and performance metrics.

    4. Revenue per Session

        Measures the revenue generated per unit of time or interaction (session). This concept originates from unit economics, which is used to understand the productivity of each user activity in generating value.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Behavioral Intensity Features**

    ---
    """)
    return


@app.cell
def _(df, safe_divide):
    # Engagement per Minute
    df['engagement_per_minute'] = safe_divide(df['engagement_score'], df['daily_session_time_minutes'])

    # Activity Intensity
    df['activity_intensity'] = safe_divide(
        df['product_views_per_day'] + df['ad_clicks_per_day'] + df['weekly_purchases'],
        df['daily_session_time_minutes']
    )

    # View-to-Purchase Pressure
    df['view_pressure'] = safe_divide(df['product_views_per_day'], df['weekly_purchases'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Engagement per Minute

        Measures the intensity of user engagement on the platform per unit of time (engagement efficiency). In theory, this is related to engagement efficiency, which is how effectively the time spent generates valuable interactions.

    2. Activity Intensity

        Measures total user activity (views, clicks, purchases) relative to usage time. This concept originates from behavioral intensity modeling, which is used to understand how active users are in each session.

    3. View-to-Purchase Pressure

        Measures the number of exposures (views) required to generate a purchase. In theory, this is related to conversion friction, where a high value indicates a bottleneck in the funnel (many views, few purchases).

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Funnel Features**

    ---
    """)
    return


@app.cell
def _(df, safe_divide):
    # Cart Efficiency
    df['cart_efficiency'] = safe_divide(df['weekly_purchases'], df['cart_items_average'])

    # Funnel Drop Rate
    df['funnel_drop'] = safe_divide(df['checkout_abandonments_per_month'], df['cart_items_average'])

    # Purchase Likelihood
    df['purchase_likelihood'] = df['purchase_conversion_rate'] * (1 - df['cart_abandonment_rate'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Cart Efficiency

        Measures how many items in a cart actually convert into purchases. In theory, this is related to conversion efficiency in the e-commerce funnel, indicating the effectiveness of the cart-to-purchase stage.

    2. Funnel Drop Rate

        Measures the proportion of failures at the checkout stage compared to the number of items added to the cart. This concept originates from funnel analysis, where this metric indicates the drop-off rate at the critical stage before conversion.

    3. Purchase Likelihood

        Estimates the probability of a purchase by combining the conversion rate and the abandonment rate. In theory, this is a simple form of probabilistic conversion modeling, which takes into account both success and failure factors in the funnel.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Customer Value Scoring**

    ---
    """)
    return


@app.cell
def _(df):
    # Composite Value Score
    df['value_score_v2'] = df['engagement_score'] * df['weekly_purchases'] * df['average_order_value']

    # Risk-Weighted Engagement
    df['safe_engagement'] = df['engagement_score'] * (1 - df['risk_score'])

    # Lifetime Proxy
    df['lifetime_value_proxy'] = df['monthly_spend'] * df['account_age_months']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Composite Value Score

        Combines engagement, purchase frequency, and transaction value to holistically represent total customer value. In theory, this is an extension of the Customer Lifetime Value (CLV) concept and the RFM model (Frequency × Monetary × Engagement proxy).

    2. Risk-Weighted Engagement

        Adjusts engagement levels for risk to more realistically reflect the quality of interactions. This concept stems from risk-adjusted metrics, where values ​​are corrected based on the probability of negative behavior (e.g., churn or fraud).

    3. Lifetime Proxy

        Estimates customer value over time by multiplying spending by the duration of the relationship (account age). In theory, this is a simplified approach to Customer Lifetime Value (CLV) used when complete historical data is unavailable.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Ratio Features**

    ---
    """)
    return


@app.cell
def _(df, safe_divide):
    # Spend per Purchase
    df['spend_per_purchase'] = safe_divide(df['monthly_spend'], df['weekly_purchases'])

    # Spend per View
    df['spend_per_view'] = safe_divide(df['monthly_spend'], df['product_views_per_day'])

    # Click Efficiency
    df['click_efficiency'] = safe_divide(df['ad_clicks_per_day'], df['ad_views_per_day'])

    # Wishlist Conversion Proxy
    df['wishlist_conversion'] = safe_divide(df['weekly_purchases'], df['wishlist_items_count'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Spend per Purchase

        Measures the average spending for each customer transaction. In theory, this is related to the Average Order Value (AOV), which is a key metric in monetization analysis.

    2. Spend per View

        Measures the economic value generated from each product interaction (view). This concept originates from the conversion value per impression, which is used to evaluate the effectiveness of exposure on revenue.

    3. Click Efficiency

        Measures the effectiveness of an ad in converting impressions (views) into clicks. In theory, this is a form of Click-Through Rate (CTR), which is a standard metric in digital marketing for measuring ad performance.

    4. Wishlist Conversion Proxy

        Measures the ability of a wishlist to generate purchases as a strong indicator of intent. This concept stems from intent-based conversion modeling, where actions such as wishlisting are considered a higher signal of purchase intent than simply viewing.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Behavior Segment Features**

    ---
    """)
    return


@app.cell
def _(df, safe_divide):
    # Impulse vs Planned
    df['impulse_ratio_v2'] = safe_divide(df['impulse_purchases_per_month'], df['weekly_purchases'])

    # Coupon Dependency
    df['coupon_dependency'] = safe_divide(df['coupon_usage_frequency'], df['monthly_spend'])

    # Exploration vs Buying
    df['exploration_ratio'] = safe_divide(df['product_views_per_day'], df['weekly_purchases'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Impulse vs. Planned

        Measuring the proportion of impulsive purchases compared to total purchases as an indicator of quick versus planned decision-making behavior. Theoretically, this relates to consumer behavior theory, specifically the distinction between impulsive buying and planned purchasing.

    2. Coupon Dependency

        Measuring how much a customer relies on discounts to make a purchase. This concept originates from price sensitivity analysis, which is used to understand the elasticity of demand to price incentives.

    3. Exploration vs. Buying

        Measuring the level of user exploration (browsing) compared to purchasing. Theoretically, this relates to conversion funnel efficiency, where a high ratio indicates high exploration but low conversion (friction in the funnel).

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Risk & Retention Features**

    ---
    """)
    return


@app.cell
def _(df, safe_divide):
    # Churn Risk Score
    df['churn_score'] = df['cart_abandonment_rate'] * (1 - df['engagement_score'])

    # Stability Score
    df['stability_score'] = safe_divide(1, df['risk_score'] + df['cart_abandonment_rate'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Churn Risk Score

        Combines abandonment rates and low engagement to estimate the likelihood of a user leaving the platform. Theoretically, this is derived from churn prediction modeling, where negative behavior (abandonment) and low engagement are key indicators of churn.

    2. Stability Score

        Measuring user stability based on a combination of low risk and low abandonment. This concept is related to risk-adjusted stability metrics, where higher scores indicate more consistent users who are more likely to be retained over the long term.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Rank Features**

    ---
    """)
    return


@app.cell
def _(df, pd):
    # Spend Rank
    df['spend_rank'] = df['monthly_spend'].rank(pct=True)

    # Engagement Rank
    df['engagement_rank'] = df['engagement_score'].rank(pct=True)

    # Value Tier
    df['value_tier'] = pd.cut(
        df['spend_rank'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Mid', 'High'],
        include_lowest=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Spend Rank

        Converts spending values ​​into relative positions (percentiles) compared to the entire population. In theory, this derives from ranking and percentile analysis, which are used to normalize data and facilitate comparisons between individuals.

    2. Engagement Rank

        Measures the relative position of a user's engagement level compared to other users in the dataset. This concept is related to relative performance measurement, which helps identify users with proportionally high or low engagement.

    3. Value Tier

        Groups users into value categories (Low, Mid, High) based on their spending percentile distribution. In theory, this is a form of quantile-based customer segmentation commonly used in business analytics for targeting and prioritization.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Interaction Features**

    ---
    """)
    return


@app.cell
def _(df):
    # Engagement × Spend
    df['engagement_spend'] = df['engagement_score'] * df['monthly_spend']

    # Risk × Spend
    df['risk_spend'] = df['risk_score'] * df['monthly_spend']

    # Activity × Value
    df['activity_value'] = df['activity_intensity'] * df['true_revenue']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Engagement × Spend

        Combines engagement level with spend value to represent customers who are not only active but also high-value. In theory, this is a form of interaction feature in feature engineering, which captures the combined effect of two variables on outcomes.

    2. Risk × Spend

        Measuring risk exposure to revenue by combining risk level and spend value. This concept originates from risk-adjusted revenue analysis, where a high value indicates a greater potential loss for high-risk customers.

    3. Activity × Value

        Combines activity intensity with revenue value to capture user productivity in generating value. In theory, this is related to behavioral value modeling, which assesses the contribution of activities to business outcomes (revenue).

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Advanced / Non-Linear Features**

    ---
    """)
    return


@app.cell
def _(df, np):
    # Log Spend
    df['log_spend'] = np.log1p(df['monthly_spend'])

    # Squared Engagement
    df['engagement_squared'] = df['engagement_score'] ** 2

    # Polynomial Interaction
    df['engagement_risk_spend'] = df['engagement_score'] * df['risk_score'] * df['monthly_spend']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Log Spend

        Performs a logarithmic transformation to reduce skewness in spending distributions, which are typically right-skewed. Theoretically, this method originates from data transformation in statistics, which is used to stabilize variance and improve model performance.

    2. Squared Engagement

        Squares engagement values ​​to capture non-linear effects, where increased engagement can have a greater impact. This concept originates from polynomial feature expansion, which is used to model non-linear relationships in data.

    3. Polynomial Interaction

        Combines three variables to capture the complex interactions between engagement, risk, and spending. Theoretically, this is a form of higher-order interaction feature, used in machine learning to improve the model's ability to capture non-linear patterns and interdependencies between variables.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Correlation Analysis**

    ---
    """)
    return


@app.cell
def _(df):
    # Resulting feature columns after engineering
    new_feature_cols = [
        'true_revenue', 'adjusted_revenue', 'revenue_efficiency', 'revenue_per_session',
        'engagement_per_minute', 'activity_intensity', 'view_pressure',
        'cart_efficiency', 'funnel_drop', 'purchase_likelihood',
        'value_score_v2', 'safe_engagement', 'lifetime_value_proxy',
        'spend_per_purchase', 'spend_per_view', 'click_efficiency', 'wishlist_conversion',
        'impulse_ratio_v2', 'coupon_dependency', 'exploration_ratio',
        'churn_score', 'stability_score',
        'spend_rank', 'engagement_rank', 'value_tier',
        'engagement_spend', 'risk_spend', 'activity_value',
        'log_spend', 'engagement_squared', 'engagement_risk_spend'
    ]

    df[new_feature_cols].head()
    return


@app.cell
def _(df):
    # Feature columns to be used for modeling or analysis

    feature_cols = [
        'ctr', 'revenue_proxy', 'spend_to_aov_ratio', 'value_score',
        'engagement_intensity', 'activity_score', 'abandonment_impact',
        'conversion_efficiency', 'impulse_ratio', 'discount_sensitivity',
        'risk_adjusted_value',

        # Additional engineered features
        'true_revenue', 'adjusted_revenue', 'revenue_efficiency', 'revenue_per_session',
        'engagement_per_minute', 'activity_intensity', 'view_pressure',
        'cart_efficiency', 'funnel_drop', 'purchase_likelihood',
        'value_score_v2', 'safe_engagement', 'lifetime_value_proxy',
        'spend_per_purchase', 'spend_per_view', 'click_efficiency', 'wishlist_conversion',
        'impulse_ratio_v2', 'coupon_dependency', 'exploration_ratio',
        'churn_score', 'stability_score',
        'spend_rank', 'engagement_rank', 'value_tier',
        'engagement_spend', 'risk_spend', 'activity_value',
        'log_spend', 'engagement_squared', 'engagement_risk_spend'
    ]

    # Ensure that only the columns that exist in the DataFrame are included in the final feature list
    feature_cols = [col for col in feature_cols if col in df.columns]
    return (feature_cols,)


@app.cell
def _(df, np):
    # Log transformation for skewed features to improve normality and reduce the impact of outliers

    log_transform_cols = [
        'monthly_spend',
        'weekly_purchases',
        'ad_views_per_day',
        'ad_clicks_per_day',
        'true_revenue'
    ]

    for transform_col in log_transform_cols:
        if transform_col in df.columns:
            df[f'log_{transform_col}'] = np.log1p(df[transform_col])
    return


@app.cell
def _(df, feature_cols, pd):
    # Correlation analysis with the target variable 'monthly_spend'

    target = 'monthly_spend'

    corr_cols = feature_cols + [target]
    corr_cols = [col for col in corr_cols if col in df.columns]

    # Convert the target variable to numeric if it's not already, coercing errors to NaN
    df[target] = pd.to_numeric(df[target], errors='coerce')

    # Select only numeric columns for correlation analysis
    numeric_corr_cols = df[corr_cols].select_dtypes(include='number').columns.tolist()

    # Calculate Pearson and Spearman correlation
    pearson_corr_1 = (
        df[numeric_corr_cols]
        .corr(method='pearson')[target]
        .sort_values(ascending=False)
    )

    spearman_corr_1 = (
        df[numeric_corr_cols]
        .corr(method='spearman')[target]
        .sort_values(ascending=False)
    )
    return pearson_corr_1, spearman_corr_1, target


@app.cell
def _(pearson_corr_1):
    # Show pearson correlation results
    pearson_corr_1
    return


@app.cell
def _(spearman_corr_1):
    # Show spearman correlation results
    spearman_corr_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Correlation analysis between the engineered features and **monthly_spend** reveals several important patterns about customer behavior. Using both Pearson (linear) and Spearman (rank-based) correlation, it becomes clear that only a subset of features meaningfully explains customer spending, while most variables show little to no relationship. This indicates that customer behavior in this dataset is not driven by single variables, but rather by combinations of multiple factors.

    The strongest relationships are observed in features such as **spend_rank**, **log_spend**, **engagement_spend**, **risk_spend**, and **engagement_risk_spend**. These features show very high correlations, suggesting that customer spending is closely tied to a combination of engagement level, risk profile, and transactional behavior. In particular, interaction features like engagement multiplied by spending or risk demonstrate that meaningful insights only emerge when multiple behavioral dimensions are considered together. Additionally, transformations like **log_spend** and ranking approaches such as **spend_rank** help reveal patterns that are not visible in raw data, especially given the flat distribution of many variables.

    A second group of features shows moderate but still meaningful relationships with spending. These include **lifetime_value_proxy**, **revenue_proxy**, **risk_adjusted_value**, and efficiency-based metrics such as **spend_per_purchase**, **spend_to_aov_ratio**, and **spend_per_view**. These features highlight that customer value is influenced by both the frequency and efficiency of transactions. Notably, several of these variables have stronger correlations in Spearman than in Pearson, indicating that their relationships with spending are not strictly linear but remain consistent in terms of ranking.

    On the other hand, the majority of features—including metrics related to marketing exposure and digital activity such as **CTR**, **activity_score**, **conversion_efficiency**, and **abandonment-related features**—show correlations close to zero. This suggests that these variables do not contribute significantly to explaining customer spending behavior. In practical terms, this means that higher engagement in terms of clicks, views, or ad exposure does not necessarily translate into higher revenue, highlighting a disconnect between activity metrics and actual customer value.

    One particularly important finding is the negative relationship observed in **coupon_dependency**, which shows a weak negative correlation in Pearson and a strong negative correlation in Spearman. This indicates that customers who rely more heavily on coupons tend to spend less overall. From a business perspective, this suggests that highly discount-driven customers may be less valuable in the long term and that aggressive discount strategies may not always lead to increased revenue.

    Overall, the analysis demonstrates that meaningful relationships in the data only become visible after applying appropriate feature engineering, particularly through interaction and transformation of variables. It also highlights that customer spending is influenced more by integrated behavioral patterns than by isolated metrics. These findings reinforce the importance of combining engagement, risk, and transactional features to better understand and segment customers, while deprioritizing variables that do not show measurable impact on spending.


    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Feature Engineering 3**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Base Helper Columns**

    ---
    """)
    return


@app.cell
def _(df):
    # Base helper columns
    df['total_activity'] = (
        df['product_views_per_day']
        + df['ad_clicks_per_day']
        + df['weekly_purchases']
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Total Activity

        Combines various forms of user interaction (views, clicks, purchases) into a single overall activity metric. In theory, this is a form of aggregate behavioral metric, used to comprehensively represent the level of user engagement in a single variable.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Value & Monetization Features**

    ---
    """)
    return


@app.cell
def _(df, np, safe_divide):
    # Purchase Power Score
    df['purchase_power'] = safe_divide(df['monthly_spend'], df['income_level'])

    # Spending Growth Proxy
    df['spend_growth'] = safe_divide(df['monthly_spend'], df['account_age_months'])

    # High Value Score
    df['high_value_score'] = (
        df['engagement_score'] * df['monthly_spend'] * (1 - df['risk_score'])
    )

    # Revenue Stability
    df['revenue_stability'] = safe_divide(df['monthly_spend'], np.sqrt(df['weekly_purchases'] + 1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Purchase Power Score

        Measuring spending power relative to income to understand customer purchasing power. In theory, this is derived from income-based spending analysis, which is used to assess the proportion of spending to financial capacity.

    2. Spending Growth Proxy

        Estimates the rate of spending growth based on the length of time a user has used the platform. This concept is related to customer lifecycle analysis, where values ​​per time period are used to understand the development of spending behavior.

    3. High Value Score

        Combining engagement, spending, and risk to identify high-value customers with high-quality interactions. In theory, this is a form of risk-adjusted customer value modeling, which assesses value while considering potential risks.

    4. Revenue Stability

        Measuring revenue stability by normalizing spending against variations in purchasing activity. This concept is related to variance normalization, where more stable values ​​indicate consistent and predictable purchasing behavior.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Behavioral Intensity Features**

    ---
    """)
    return


@app.cell
def _(df):
    # Deep Engagement Score
    df['deep_engagement'] = (
        df['engagement_score']
        * df['daily_session_time_minutes']
        * df['total_activity']
    )

    # Active Buyer Score
    df['active_buyer'] = df['weekly_purchases'] * df['engagement_score']

    # Attention Score
    df['attention_score'] = df['daily_session_time_minutes'] * df['product_views_per_day']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Deep Engagement Score

        Combines engagement, session duration, and total activity to represent user engagement in a deep and comprehensive way. In theory, this is a form of multi-dimensional engagement modeling, capturing the intensity of interactions across various aspects of user behavior.

    2. Active Buyer Score

        Measuring the combination of purchase frequency and engagement level to identify active and valuable users. This concept stems from behavioral segmentation, specifically in identifying high-frequency and high-engagement customers.

    3. Attention Score

        Measuring the level of user attention on a platform based on time spent and number of interactions (views). In theory, this relates to the attention economy, where user time and focus are key indicators of the value of interactions on digital platforms.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Funnel Features**

    ---
    """)
    return


@app.cell
def _(df, safe_divide):
    # Funnel Efficiency Score
    df['funnel_efficiency'] = safe_divide(
        df['weekly_purchases'],
        df['product_views_per_day'] + df['cart_items_average']
    )

    # Checkout Strength
    df['checkout_strength'] = (
        (1 - df['cart_abandonment_rate']) * df['weekly_purchases']
    )

    # Conversion Pressure
    df['conversion_pressure'] = safe_divide(
        df['product_views_per_day'],
        df['purchase_conversion_rate']
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Funnel Efficiency Score

        Measuring the overall effectiveness of the funnel by comparing the number of purchases to the total initial exposures (views) and purchase intent (cart). In theory, this is derived from conversion funnel analysis, which assesses how efficiently users move from awareness to purchase.

    2. Checkout Strength

        Combining checkout success rate with purchase frequency to reflect the strength of the final stage of the funnel. This concept is related to conversion success modeling, where success in the final stage (checkout) is the primary determinant of revenue.

    3. Conversion Pressure

        Measuring the number of exposures (views) required to generate a conversion, as an indicator of pressure in the conversion process. In theory, this is related to conversion friction, where a high value indicates a high effort (views) required to achieve a single conversion.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Risk & Retention Features**

    ---
    """)
    return


@app.cell
def _(df):
    # Safe Revenue Score
    df['safe_revenue'] = df['monthly_spend'] * (1 - df['risk_score'])

    # Risk Exposure
    df['risk_exposure'] = df['risk_score'] * df['cart_abandonment_rate']

    # Retention Strength
    df['retention_strength'] = df['engagement_score'] * (1 - df['cart_abandonment_rate'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Safe Revenue Score

        Adjusts revenue values ​​according to risk levels to obtain safer and more realistic revenue estimates. In theory, this stems from risk-adjusted revenue modeling, which reduces values ​​based on the probability of potential losses.

    2. Risk Exposure

        Measuring the level of risk exposure by combining general user risk and negative behavior in the funnel (abandonment). This concept is related to risk aggregation, where multiple risk sources are combined to assess the total impact on the business.

    3. Retention Strength

        Measuring user retention strength based on engagement levels and low abandonment behavior. In theory, this stems from retention modeling, where high engagement and low attrition are key indicators of user retention.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Behavior Segment Features**

    ---
    """)
    return


@app.cell
def _(df, safe_divide):
    # Premium Behavior Score
    df['premium_behavior'] = safe_divide(
        df['monthly_spend'],
        df['coupon_usage_frequency'] + 1
    )

    # Discount Dependency Score
    df['discount_score'] = safe_divide(df['coupon_usage_frequency'], df['monthly_spend'])

    # Impulse Behavior Score
    df['impulse_score'] = df['impulse_purchases_per_month'] * (1 - df['engagement_score'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Premium Behavior Score

        Measuring a customer's tendency to make high-value purchases without relying on discounts. In theory, this is related to premium customer segmentation, where customers with a high willingness to pay tend to be less price sensitive.

    2. Discount Dependency Score

        Measuring the level of customer dependence on coupon use in shopping activities. This concept originates from price sensitivity analysis, which is used to understand the elasticity of demand to price incentives.

    3. Impulse Behavior Score

        Combining impulsive buying with low engagement to identify unplanned purchasing behavior. In theory, this is related to impulse buying behavior, where purchasing decisions occur spontaneously without deep involvement.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Efficiency Features**

    ---
    """)
    return


@app.cell
def _(df, safe_divide):
    # Spend Efficiency
    df['spend_efficiency'] = safe_divide(df['monthly_spend'], df['total_activity'])

    # Engagement Efficiency
    df['engagement_eff'] = safe_divide(df['engagement_score'], df['total_activity'])

    # Revenue per Click
    df['revenue_per_click'] = safe_divide(df['monthly_spend'], df['ad_clicks_per_day'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Spend Efficiency

        Measures the amount of revenue generated for each unit of user activity. Theoretically, this is related to the efficiency ratio in unit economics, which is used to assess the productivity of activities relative to financial output.

    2. Engagement Efficiency

        Measures the level of engagement generated per unit of user activity. This concept stems from engagement normalization, which aims to fairly compare the quality of interactions between users with different levels of activity.

    3. Revenue per Click

        Measures the monetization value generated from each user click. Theoretically, this is related to the value per interaction in digital marketing, which is used to evaluate the effectiveness of clicks in generating revenue.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Rank Features**

    ---
    """)
    return


@app.cell
def _(df):
    # Spend Percentile
    df['spend_percentile'] = df['monthly_spend'].rank(pct=True)

    # Engagement Percentile
    df['engagement_percentile'] = df['engagement_score'].rank(pct=True)

    # Risk Percentile
    df['risk_percentile'] = df['risk_score'].rank(pct=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Spend Percentile

        Converts spending values ​​into relative positions within a population distribution to understand spending levels comparatively. In theory, this is derived from percentile ranking in statistics, which is used for data normalization and segmentation.

    2. Engagement Percentile

        Measures the relative position of a user's engagement compared to all users in a dataset. This concept is related to relative performance measurement, which allows the identification of users with proportionally high or low engagement.

    3. Risk Percentile

        Describes the relative position of a user's risk level within the overall population. In theory, this is part of risk scoring and ranking, which is used to group users based on their risk level in business decision-making.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Interaction Features**

    ---
    """)
    return


@app.cell
def _(df):
    # Triple Interaction
    df['triple_interaction'] = (
        df['engagement_score'] * df['risk_score'] * df['monthly_spend']
    )

    # Activity × Spend
    df['activity_spend'] = df['total_activity'] * df['monthly_spend']

    # Risk × Engagement × Activity
    df['risk_engagement_activity'] = (
        df['risk_score'] * df['engagement_score'] * df['total_activity']
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Triple Interaction

        Menggabungkan engagement, risk, dan spending untuk menangkap interaksi kompleks antar variabel utama yang mempengaruhi nilai customer. Secara teori, ini merupakan bentuk higher-order interaction feature, yang digunakan dalam machine learning untuk memodelkan hubungan non-linear dan interdependensi variabel.

    2. Activity × Spend

        Mengukur kontribusi aktivitas user terhadap nilai finansial yang dihasilkan. Konsep ini berkaitan dengan behavioral value modeling, di mana aktivitas dianggap sebagai driver utama revenue.

    3. Risk × Engagement × Activity

        Menggabungkan risiko, keterlibatan, dan aktivitas untuk mengevaluasi kualitas interaksi user secara menyeluruh. Secara teori, ini merupakan bentuk multi-factor interaction modeling, yang digunakan untuk menangkap keseimbangan antara value creation dan risk exposure.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Advanced / Non-Linear Features**

    ---
    """)
    return


@app.cell
def _(df, np):
    # Log Interaction
    df['log_interaction'] = np.log1p(df['engagement_score'] * df['monthly_spend'])

    # Squared Spend
    df['squared_spend'] = df['monthly_spend'] ** 2

    # Exponential Engagement
    df['exp_engagement'] = np.exp(np.clip(df['engagement_score'], None, 10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Log Interaction

        Performs a logarithmic transformation on the interaction between engagement and spending to reduce skewness and stabilize the distribution. Theoretically, this is derived from the log transformation in statistics, which is used to handle data with non-normal distributions and large scales.

    2. Squared Spend

        Squares the spending value to capture non-linear effects, where large increases in spending can have a larger impact. This concept comes from polynomial feature expansion, which is used to model non-linear relationships in data.

    3. Exponential Engagement

        Applies an exponential function to amplify differences at high engagement values. Theoretically, this is related to the non-linear transformation, which is used to emphasize the effects of sharp increases in certain variables in the model.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  **Segmentation Features**

    ---
    """)
    return


@app.cell
def _(df, pd):
    # Value Tier
    df['value_tier_v2'] = pd.cut(
        df['spend_percentile'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )

    # Risk Tier
    df['risk_tier'] = pd.cut(
        df['risk_percentile'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk'],
        include_lowest=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Value Tier

        Mengelompokkan customer ke dalam kategori nilai berdasarkan distribusi persentil spending untuk memudahkan segmentasi. Secara teori, ini merupakan bentuk quantile-based segmentation, yang umum digunakan dalam analisis bisnis untuk mengidentifikasi kelompok customer bernilai rendah, menengah, dan tinggi.

    2. Risk Tier

        Mengelompokkan customer berdasarkan tingkat risiko relatif terhadap populasi. Konsep ini berasal dari risk segmentation, yang digunakan untuk membedakan strategi penanganan antara user berisiko rendah hingga tinggi dalam pengambilan keputusan bisnis.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Advanced Business Features**

    ---
    """)
    return


@app.cell
def _(df, safe_divide):
    # Customer Lifetime Index
    df['cl_index'] = df['monthly_spend'] * df['age'] * df['engagement_score']

    # Profitability Proxy
    coupon_dependency = safe_divide(df['coupon_usage_frequency'], df['monthly_spend'])
    df['profit_proxy'] = df['monthly_spend'] * (1 - coupon_dependency.fillna(0))

    # Strategic Value Score
    df['strategic_value'] = safe_divide(
        df['monthly_spend'] * df['engagement_score'],
        df['risk_score']
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    1. Customer Lifetime Index

        Combines spending, age (as a lifecycle proxy), and engagement to estimate a customer's long-term value. In theory, this is a simplified approach to Customer Lifetime Value (CLV) that integrates the dimensions of time, activity, and monetization.

    2. Profitability Proxy

        Estimates profitability by adjusting revenue for reliance on discounts. This concept originates from margin-based analysis, where the use of incentives (coupons) is considered to reduce net profit.

    3. Strategic Value Score

        Measures a customer's strategic value by combining engagement and revenue, then adjusting for risk. In theory, this is a form of risk-adjusted value modeling, used to prioritize high-value and low-risk customers.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Correlation Analysis**

    ---
    """)
    return


@app.cell
def _(df, np):
    # Cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)


    # Display the new features
    new_feature_3_cols = [
        'total_activity',
        'purchase_power', 'spend_growth', 'high_value_score', 'revenue_stability',
        'deep_engagement', 'active_buyer', 'attention_score',
        'funnel_efficiency', 'checkout_strength', 'conversion_pressure',
        'safe_revenue', 'risk_exposure', 'retention_strength',
        'premium_behavior', 'discount_score', 'impulse_score',
        'spend_efficiency', 'engagement_eff', 'revenue_per_click',
        'spend_percentile', 'engagement_percentile', 'risk_percentile',
        'triple_interaction', 'activity_spend', 'risk_engagement_activity',
        'log_interaction', 'squared_spend', 'exp_engagement',
        'value_tier_v2', 'risk_tier','cl_index', 'profit_proxy', 'strategic_value'
    ]

    df[new_feature_3_cols].head()
    return


@app.cell
def _(df):
    # Feature columns to be used for modeling or analysis, including the new engineered features
    feature_cols_2 = [
        'ctr', 'revenue_proxy', 'spend_to_aov_ratio', 'value_score',
        'engagement_intensity', 'activity_score', 'abandonment_impact',
        'conversion_efficiency', 'impulse_ratio', 'discount_sensitivity',
        'risk_adjusted_value',

        # Additional engineered features 1
        'true_revenue', 'adjusted_revenue', 'revenue_efficiency', 'revenue_per_session',
        'engagement_per_minute', 'activity_intensity', 'view_pressure',
        'cart_efficiency', 'funnel_drop', 'purchase_likelihood',
        'value_score_v2', 'safe_engagement', 'lifetime_value_proxy',
        'spend_per_purchase', 'spend_per_view', 'click_efficiency', 'wishlist_conversion',
        'impulse_ratio_v2', 'coupon_dependency', 'exploration_ratio',
        'churn_score', 'stability_score',
        'spend_rank', 'engagement_rank', 'value_tier',
        'engagement_spend', 'risk_spend', 'activity_value',
        'log_spend', 'engagement_squared', 'engagement_risk_spend'

        # Additional engineered features 2
        'total_activity',
        'purchase_power', 'spend_growth', 'high_value_score', 'revenue_stability',
        'deep_engagement', 'active_buyer', 'attention_score',
        'funnel_efficiency', 'checkout_strength', 'conversion_pressure',
        'safe_revenue', 'risk_exposure', 'retention_strength',
        'premium_behavior', 'discount_score', 'impulse_score',
        'spend_efficiency', 'engagement_eff', 'revenue_per_click',
        'spend_percentile', 'engagement_percentile', 'risk_percentile',
        'triple_interaction', 'activity_spend', 'risk_engagement_activity',
        'log_interaction', 'squared_spend', 'exp_engagement',
        'value_tier_v2', 'risk_tier','cl_index', 'profit_proxy', 'strategic_value'
    ]

    # Ensure that only the columns that exist in the DataFrame are included in the final feature list
    feature_cols_2 = [col for col in feature_cols_2 if col in df.columns]
    return


@app.cell
def _(df, np):
    # Log transformation for skewed features in the new engineered features to improve normality and reduce the impact of outliers
    log_transform_cols_2 = [
        # Base features
        'monthly_spend',
        'weekly_purchases',
        'ad_views_per_day',
        'ad_clicks_per_day',
        'true_revenue',

        # Value / Revenue
        'adjusted_revenue',
        'lifetime_value_proxy',
        'cl_index',
        'strategic_value',

        # Activity / Engagement
        'total_activity',
        'activity_spend',
        'deep_engagement',
        'attention_score',

        # Interactions
        'engagement_spend',
        'risk_spend',
        'triple_interaction',

        # Efficiency / Ratios
        'spend_per_purchase',
        'spend_efficiency',
        'revenue_per_click',

        # Additional engineered features that may have skewed distributions
        'high_value_score',
        'safe_revenue',
        'profit_proxy'
    ]

    # Ensure that only the columns that exist in the DataFrame are included in the log transformation list
    log_transform_cols_2 = [col for col in log_transform_cols_2 if col in df.columns]

    # Apply log transformation to the specified columns
    for transform_col_2 in log_transform_cols_2:
        df[f'log_{transform_col_2}'] = np.log1p(df[transform_col_2])
    return


@app.cell
def _(df, feature_cols, pd, target):
    # CORRELATION ANALYSIS
    corr_cols_2 = feature_cols + [target]
    corr_cols_2 = [col for col in corr_cols_2 if col in df.columns]

    # Ensure the target variable is numeric for correlation analysis
    df[target] = pd.to_numeric(df[target], errors='coerce')

    # Select only numeric columns for correlation analysis
    numeric_corr_cols_2 = df[corr_cols_2].select_dtypes(include='number').columns.tolist()

    # Calculate Pearson and Spearman correlation
    pearson_corr_2 = (
        df[numeric_corr_cols_2]
        .corr(method='pearson')[target]
        .sort_values(ascending=False)
    )

    spearman_corr_2 = (
        df[numeric_corr_cols_2]
        .corr(method='spearman')[target]
        .sort_values(ascending=False)
    )
    return


@app.cell
def _(pearson_corr_1):
    # Show updated Pearson correlation results
    pearson_corr_1
    return


@app.cell
def _(spearman_corr_1):
    # Show spearman correlation results
    spearman_corr_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The correlation analysis between engineered features and **monthly_spend** reveals a clear distinction between variables that meaningfully explain customer spending and those that do not. Using both Pearson and Spearman correlation, it becomes evident that strong relationships only emerge from a specific group of features, particularly those derived from transformations and interactions.

    At the top level, features such as **spend_rank**, **log_spend**, **engagement_spend**, **risk_spend**, and **engagement_risk_spend** show very strong correlations with monthly spending. These results indicate that customer value is highly associated with a combination of engagement level, risk profile, and spending behavior. In particular, interaction features—such as the combination of engagement and spending or risk and spending—demonstrate that meaningful patterns are not captured by single variables alone, but rather by how multiple behavioral factors interact with each other. The use of transformations like logarithmic scaling and ranking also proves effective in revealing patterns that were not visible in the raw data.

    A second group of features shows moderate but still important relationships with spending. These include **lifetime_value_proxy**, **revenue_proxy**, **risk_adjusted_value**, and efficiency-related features such as **spend_per_purchase**, **spend_to_aov_ratio**, and **spend_per_view**. These variables suggest that customer spending is influenced not only by total activity but also by how efficiently users convert their actions into transactions. The fact that several of these features have stronger correlations in Spearman than in Pearson further indicates that the relationships are not strictly linear but still consistent when viewed in terms of ranking.

    In contrast, the majority of other features—including metrics related to marketing exposure, user activity, and funnel behavior—show correlations that are extremely close to zero. Features such as **CTR**, **activity_score**, **conversion_efficiency**, **abandonment-related metrics**, and various derived scores do not exhibit any meaningful relationship with monthly spending. This suggests that higher levels of activity, engagement with ads, or even certain behavioral indicators do not necessarily translate into increased revenue. In practical terms, this highlights a disconnect between operational metrics and actual customer value.

    One of the most significant findings is the negative relationship observed in **coupon_dependency**. While the Pearson correlation shows a mild negative relationship, the Spearman correlation reveals a much stronger negative association. This indicates that customers who rely heavily on coupons tend to spend less overall. From a business perspective, this suggests that discount-driven customers may be less valuable and that heavy reliance on promotional strategies may not lead to sustainable revenue growth.

    Overall, the analysis demonstrates that meaningful relationships within the dataset only become visible after applying appropriate feature engineering, especially through interaction and transformation of variables. It also highlights that customer spending is driven more by integrated behavioral patterns—combining engagement, risk, and transaction characteristics—rather than by isolated metrics. These findings emphasize the importance of focusing on high-impact features while deprioritizing variables that do not contribute to explaining customer value.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Heatmap Correlation**

    ---
    """)
    return


@app.cell
def _(df, np, output_dir, plt, sns):
    # Visualize the correlation matrix for the numeric features
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_df.dropna(axis=1, how='all', inplace=True)

    # Calculate the correlation matrix using Pearson method
    corr_matrix = numeric_df.corr(method='pearson')

    # Filter the correlation matrix to show only correlations above a certain threshold for better visualization
    threshold = 0.5
    filtered_corr = corr_matrix.copy()
    filtered_corr[filtered_corr.abs() < threshold] = np.nan

    # Create a mask to display only the lower triangle of the correlation matrix
    mask = np.triu(np.ones_like(filtered_corr, dtype=bool))

    plt.figure(figsize=(18, 14))
    sns.set_style("white")

    sns.heatmap(
        filtered_corr,
        mask=mask,
        cmap='coolwarm',
        center=0,
        linewidths=0.3,
        square=False,
        cbar_kws={'shrink': 0.8}
    )

    # Save the figure for the current batch
    filename_correlation = f"{output_dir}/correlation_heatmap.png"
    plt.savefig(filename_correlation, dpi=300, bbox_inches='tight')


    plt.title(f"Filtered Correlation Heatmap (|corr| >= {threshold})", fontsize=14)
    plt.tight_layout()
    plt.show()
    return (corr_matrix,)


@app.cell
def _(corr_matrix):
    # Convert the correlation matrix into a long format table for easier analysis of feature pairs
    corr_table = (
        corr_matrix
        .stack()
        .reset_index()
    )

    corr_table.columns = ['feature_1', 'feature_2', 'correlation']

    # Remove self-correlation
    corr_table = corr_table[corr_table['feature_1'] != corr_table['feature_2']]

    # Remove duplicate pairs (A-B and B-A)
    corr_table['pair'] = corr_table.apply(
        lambda x: tuple(sorted([x['feature_1'], x['feature_2']])), axis=1
    )

    corr_table = corr_table.drop_duplicates('pair').drop(columns='pair')

    # sort by absolute correlation
    corr_table = corr_table.reindex(
        corr_table['correlation'].abs().sort_values(ascending=False).index
    )

    # Display the top correlated feature pairs
    corr_table
    return (corr_table,)


@app.cell
def _(corr_table):
    # Filter the correlation table to show only pairs with high correlation (e.g., above 0.7)
    threshold_high = 0.7

    high_corr = corr_table[
        corr_table['correlation'].abs() >= threshold_high   
    ].sort_values(by='correlation', key=lambda x: abs(x), ascending=False)

    # Display the highly correlated feature pairs
    high_corr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The analysis reveals several deeper insights that are important for understanding customer behavior from both a statistical and business perspective:
    - First, the dataset does not exhibit a strong natural signal. Most raw features show correlations close to zero with **monthly spending**, which indicates that meaningful patterns are not immediately visible. In other words, the data does not naturally explain customer value on its own. Instead, insights only emerge after combining variables or applying feature engineering. From a business standpoint, this means that relying on a single metric—such as clicks or views—is not sufficient to evaluate customer value. A more holistic approach that combines multiple behavioral indicators is required.

    - Second, customers with high performance tend to be consistent across all value-related metrics. Features such as **monthly_spend**, **revenue_proxy**, and **lifetime_value_proxy** are strongly correlated with each other, showing that high-value customers are consistently identified regardless of the metric used. This suggests that valuable customers form a clearly distinguishable segment. From a business perspective, this makes prioritization easier, as these high-value customers should be the primary focus for retention and long-term engagement strategies.

    - Third, there is a strong presence of multicollinearity in the dataset. Many features are simply transformations of the same underlying variable, such as **monthly_spend**, **spend_rank**, and **log_spend**, or **engagement_score**, **engagement_rank**, and **engagement_percentile**. These features do not add new information but instead represent the same concept in different forms. As a result, having too many similar features can make the analysis more complex and less interpretable. Reducing redundant features is therefore essential to maintain clarity and avoid misleading conclusions.

    - Fourth, customer behavior can be grouped into three main layers: value, engagement, and risk. The value layer includes spending and revenue-related metrics, the engagement layer reflects user activity such as views and sessions, and the risk layer captures behaviors like cart abandonment and returns. Importantly, these layers do not always align. A customer may be highly active but generate little revenue, or may generate high revenue while also showing risky behavior. This separation highlights the need for multi-dimensional analysis when evaluating customers.

    - Fifth, marketing exposure alone does not drive revenue. Metrics such as **CTR**, ad views, and clicks show almost no correlation with spending. This indicates that simply increasing visibility or engagement does not guarantee higher revenue. From a business perspective, this reinforces the idea that marketing strategies should prioritize conversion and value generation rather than just reach or activity.

    - Sixth, there are clear signs of over-engineering in the feature set. The correlation heatmap shows many features with extremely high correlations (close to 1), indicating redundancy. While feature engineering is necessary to uncover patterns, excessive or redundant features can introduce noise, bias the analysis, and make interpretation more difficult. A more focused and selective approach to feature creation is therefore recommended.

    - Finally, the relationship between variables appears to be stronger when analyzed using ranking-based methods (Spearman) rather than linear methods (Pearson). This suggests that many relationships in the data are not strictly linear but remain consistent in terms of ordering. In practical terms, this means that customer segmentation is better approached using relative measures such as percentiles or rankings rather than fixed thresholds.

    Overall, these insights highlight that customer behavior is complex and multi-dimensional. Understanding it requires combining multiple features, reducing redundancy, and focusing on meaningful behavioral patterns rather than isolated metrics.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Relevance-based Column Filtering**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Feature Selection**

    ---
    """)
    return


@app.cell
def _():
    # Remove Columns Based on Correlation and Redundancy

    drop_cols = [
        # 1. Redundant Spend
        'spend_rank',
        'spend_percentile',
        'log_spend',
        'log_monthly_spend',

        # 2. Redundant Engagement
        'engagement_rank',
        'engagement_percentile',
        'engagement_squared',
        'exp_engagement',

        # 3. Redundant Value
        'adjusted_revenue',
        'true_revenue',
        'value_score',
        'value_score_v2',

        # 4. Noise
        'activity_score',
        'activity_intensity',
        'conversion_efficiency',
        'click_efficiency',
        'funnel_drop',
        'wishlist_conversion',
        'cart_efficiency',

        # 5. Over-complex
        'engagement_risk_spend',
        'triple_interaction',
        'strategic_value',
        'log_interaction',
        'squared_spend',

        # 6. Low Correlation
        'impulse_ratio',
        'impulse_ratio_v2',
        'exploration_ratio',
        'view_pressure'
    ]
    return (drop_cols,)


@app.cell
def _(df, drop_cols):
    # Handle log-transformed columns by identifying them dynamically
    log_cols = [col for col in df.columns if col.startswith('log_')]

    # Combine the predefined drop columns with the dynamically identified log columns
    drop_cols_extended = drop_cols + log_cols

    # Ensure that only the columns that exist in the DataFrame are included in the final drop list
    drop_cols_final = [col for col in drop_cols_extended if col in df.columns]

    # Create a cleaned DataFrame by dropping the identified columns
    df_clean = df.drop(columns=drop_cols_final)
    print(f"Number of columns before: {df.shape[1]}")
    print(f"Number of columns after: {df_clean.shape[1]}")
    print(f"Number of columns dropped: {len(drop_cols_final)}")
    return (df_clean,)


@app.cell
def _(df_clean):
    # Display the cleaned DataFrame with the remaining features
    df_clean
    return


@app.cell
def _(df):
    # Display the remaining columns after dropping
    df.columns.tolist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    The data that has been cleaned and after meeting business standards will be used for further analysis.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Business Relevance Selection**

    ---
    """)
    return


@app.cell
def _(df):
    # Define the list of columns to be selected for the final dataset, ensuring they are relevant with customer behavior and not redundant
    selected_cols = [
        # Core Value
        'monthly_spend',
        'weekly_purchases',
        'average_order_value',

        # Demographic
        'age',
        'gender',
        'country',
        'income_level',
        'income_group',
        'employment_status',
        'education_level',
        'household_size',

        # Engagement
        'daily_session_time_minutes',
        'product_views_per_day',
        'app_usage_frequency',
        'wishlist_items_count',
        'engagement_score',

        # Marketing
        'ad_views_per_day',
        'ad_clicks_per_day',
        'notification_response_rate',
        'coupon_usage_frequency',
        'ctr',
        'coupon_dependency',

        # Risk
        'cart_abandonment_rate',
        'checkout_abandonments_per_month',
        'return_rate',
        'return_frequency',
        'risk_score',

        # Loyalty
        'loyalty_program_member',
        'premium_subscription',

        # Recency
        'account_age_months',
        'last_purchase_date',
        'purchase_month',
        'purchase_day',
        'day_of_week',

        # Value Proxy
        'revenue_proxy',
        'lifetime_value_proxy',
        'risk_adjusted_value',

        # Efficiency
        'spend_per_purchase',
        'spend_to_aov_ratio',
        'spend_per_view',

        # Interaction
        'engagement_spend',
        'risk_spend'
    ]

    # Column filtering to ensure only existing columns are selected
    selected_cols_final = [col for col in selected_cols if col in df.columns]

    # New DataFrame with selected columns
    df_selected = df[selected_cols_final].copy()

    # Display the number of columns before and after selection
    print(f"Initial number of columns: {df.shape[1]}")
    print(f"Number of selected columns: {df_selected.shape[1]}")

    df_selected.head()
    return (df_selected,)


@app.cell
def _(df_selected):
    # Display the list of columns in the final selected DataFrame
    df_selected.columns.tolist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    | Category        | Column                          | Priority | Analysis Purpose        | Reason for Retention                    |
    | --------------- | ------------------------------- | -------- | ----------------------- | --------------------------------------- |
    | **Core Value**  | monthly_spend                   | 🔥 High  | Measure customer value  | Main target, foundation of all analysis |
    |                 | weekly_purchases                | 🔥 High  | Purchase frequency      | Determines transaction intensity        |
    |                 | average_order_value             | 🔥 High  | Value per transaction   | Explains spending quality               |
    | **Demographic** | age                             | ⚡ High   | Customer profiling      | Age segmentation                        |
    |                 | gender                          | ⚡ High   | Customer profiling      | Gender comparison                       |
    |                 | country                         | ⚡ High   | Geographic distribution | Market segmentation                     |
    |                 | income_level / income_group     | ⚡ High   | Purchasing power        | Correlates with spending                |
    |                 | employment_status               | ⚡ Medium | Economic profile        | Additional insight                      |
    |                 | education_level                 | ⚡ Medium | Social profile          | Additional segmentation                 |
    |                 | household_size                  | ⚡ Medium | Family structure        | Behavioral insight                      |
    | **Engagement**  | daily_session_time_minutes      | 🔥 High  | User activity           | Measures interaction time               |
    |                 | product_views_per_day           | 🔥 High  | Browsing activity       | Indicates interest level                |
    |                 | app_usage_frequency             | ⚡ High   | App activity            | Usage consistency                       |
    |                 | wishlist_items_count            | ⚡ Medium | Purchase intent         | Indicates interest                      |
    |                 | engagement_score                | 🔥 High  | Engagement summary      | Represents overall activity             |
    | **Marketing**   | ad_views_per_day                | ⚡ High   | Ad exposure             | Awareness measurement                   |
    |                 | ad_clicks_per_day               | ⚡ High   | Ad response             | Ad engagement                           |
    |                 | notification_response_rate      | ⚡ High   | Notification response   | Marketing effectiveness                 |
    |                 | coupon_usage_frequency          | ⚡ High   | Discount usage          | Price sensitivity                       |
    |                 | ctr                             | ⚡ Medium | Ad effectiveness        | Click-through rate                      |
    |                 | coupon_dependency               | 🔥 High  | Discount dependency     | Indicates low-value customers           |
    | **Risk**        | cart_abandonment_rate           | 🔥 High  | Transaction risk        | Indicates friction in purchase          |
    |                 | checkout_abandonments_per_month | ⚡ High   | Funnel drop-off         | Behavioral risk                         |
    |                 | return_rate / return_frequency  | 🔥 High  | Return behavior         | Product mismatch indicator              |
    |                 | risk_score                      | 🔥 High  | Customer risk           | Summary risk indicator                  |
    | **Loyalty**     | loyalty_program_member          | 🔥 High  | Loyalty segmentation    | Compare behavior across groups          |
    |                 | premium_subscription            | 🔥 High  | Premium segmentation    | Differentiates value tiers              |
    | **Recency**     | account_age_months              | ⚡ High   | Customer lifecycle      | Duration as customer                    |
    |                 | last_purchase_date              | ⚡ High   | Recent activity         | Detect inactive users                   |
    |                 | purchase_month                  | ⚡ Medium | Seasonality             | Time-based patterns                     |
    |                 | purchase_day                    | ⚡ Medium | Daily behavior          | Behavioral timing                       |
    |                 | day_of_week                     | ⚡ Medium | Weekend vs weekday      | Shopping timing behavior                |
    | **Value Proxy** | revenue_proxy                   | 🔥 High  | Revenue estimation      | Strong correlation with spending        |
    |                 | lifetime_value_proxy            | 🔥 High  | Long-term value         | Customer segmentation                   |
    |                 | risk_adjusted_value             | 🔥 High  | Value adjusted by risk  | Customer prioritization                 |
    | **Efficiency**  | spend_per_purchase              | ⚡ High   | Transaction efficiency  | Value per purchase                      |
    |                 | spend_to_aov_ratio              | ⚡ High   | Spending comparison     | Buying pattern                          |
    |                 | spend_per_view                  | ⚡ Medium | Browsing efficiency     | Conversion insight                      |
    | **Interaction** | engagement_spend                | 🔥 High  | Engagement × value      | Strongest behavioral feature            |
    |                 | risk_spend                      | 🔥 High  | Risk × value            | Captures complex behavior               |


    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Make New DataFrame**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Checking Data**

    ---
    """)
    return


@app.cell
def _(df_selected):
    # Check for missing values in the selected DataFrame
    df_selected.isnull().sum()
    return


@app.cell
def _(df):
    # Check for missing values in the original DataFrame to understand the extent of missing data
    df.isnull().sum()
    return


@app.cell
def _(df, df_selected):
    # # Depending on the amount of missing data, we can decide to drop rows with missing values or impute them. For simplicity, we'll drop rows with missing values in the selected DataFrame.
    df_selected_save = df_selected.dropna().copy()
    df_save = df.dropna().copy()
    return df_save, df_selected_save


@app.cell
def _(df_selected_save):
    df_selected_save
    return


@app.cell
def _(df_selected_save):
    df_selected_save.info()
    return


@app.cell
def _(df_save):
    df_save
    return


@app.cell
def _(df_save):
    df_save.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Because of division by zero, the result will be considered a missing value.

    ---
    """)
    return


@app.cell
def _():
    # # Save selected dataset
    # selected_file = f"{output_dir}/customer_data_selected.csv"
    # df_selected_save.to_csv(selected_file, index=False)

    # # Save full cleaned dataset
    # full_file = f"{output_dir}/customer_data_full.csv"
    # df_save.to_csv(full_file, index=False)

    # print(f"Selected dataset saved to: {selected_file}")
    # print(f"Full dataset saved to: {full_file}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insight:**

    Overall, the selected columns are sufficient to address all the objectives of the **02_eda_shopper_behavior.ipynb** notebook. This is because each group of columns directly represents the key aspects being analyzed, including customer profile, purchasing behavior, engagement, marketing, risk, loyalty, and recency.

    To understand customer profiles and characteristics, demographic columns such as age, gender, country, and income level provide a clear picture of who the users are in the dataset. These variables allow for distribution analysis and identification of key customer segments based on fundamental attributes. Although these features may not always show a direct correlation with spending, they remain essential for descriptive analysis and initial segmentation.

    In analyzing purchasing behavior, columns such as **monthly_spend**, **weekly_purchases**, and **average_order_value** serve as the core foundation. These variables are sufficient to explain how customers transact, both in terms of frequency and value. With additional derived features like **spend_per_purchase** and **spend_to_aov_ratio**, the analysis becomes more insightful, enabling a deeper understanding of whether customers tend to purchase frequently in small amounts or infrequently with higher transaction values.

    To identify engagement levels, columns such as **daily_session_time_minutes**, **product_views_per_day**, and **engagement_score** are adequate to describe how active customers are on the platform. These features allow comparisons between highly active and less active users, and help determine whether higher engagement translates into increased spending.

    From a marketing perspective, columns such as **ad_views_per_day**, **ad_clicks_per_day**, **coupon_usage_frequency**, and **notification_response_rate** can be used to evaluate campaign effectiveness. With additional metrics like **CTR** and **coupon_dependency**, the analysis can reveal whether interactions with ads or the use of discounts actually influence spending. The results suggest that not all marketing activities contribute directly to revenue generation.

    To detect risk behavior, columns such as **cart_abandonment_rate**, **return_rate**, and **risk_score** provide strong indicators of potential issues in the customer journey. These features enable the identification of high-risk customers, such as those who frequently abandon carts or return products, which may ultimately lead to churn.

    Comparative analysis between loyal and non-loyal customers, as well as premium and non-premium users, can be effectively conducted using status indicators like **loyalty_program_member** and **premium_subscription**. By comparing key metrics such as spending and engagement across these groups, it is possible to evaluate whether loyalty programs or premium offerings significantly influence customer behavior.

    For recency analysis, columns such as **account_age_months** and **last_purchase_date** are sufficient to distinguish between active and inactive customers. This is important for understanding the customer lifecycle and identifying potential churn based on the timing of their last activity.

    When exploring relationships between variables, features such as **monthly_spend**, **revenue_proxy**, **lifetime_value_proxy**, **engagement_spend**, and **risk_spend** are sufficient to identify the main drivers of customer value. These engineered features enable more meaningful correlation analysis compared to raw variables, as they capture interactions between different behavioral aspects.

    Overall, the selected combination of columns covers all necessary dimensions to produce the expected outputs, ranging from descriptive statistics to key findings. There is no need to introduce additional features, as including too many variables can make the analysis less focused and harder to interpret.

    In conclusion, these columns are adequate to fulfill all EDA objectives, as they effectively explain who the customers are, how they behave, what drives their value, and how they can be segmented for further analysis such as customer segmentation and churn prediction.

    ---
    """)
    return


if __name__ == "__main__":
    app.run()
