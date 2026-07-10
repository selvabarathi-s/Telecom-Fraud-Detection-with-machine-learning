# Telecom Fraud Detection

**call_id**                        - Unique identifier for each call record (e.g. CALL0000001)

**caller_age_days**                - How long the caller's number/account has
existed, in days. Fraudulent numbers tend to be newer.

**calls_per_day**                 - Average number of calls made by this caller per day. Spammers/fraudsters often call at high volume.

**call_duration_sec**             - Duration of this specific call, in seconds.

**avg_call_duration_sec**          - This caller's average call duration across all their calls.

**unique_receivers_24h**           - Number of distinct people this caller contacted in the last 24 hours. A high count can indicate mass dialing/spam.

**receiver_block_rate**            - Percentage of receivers who have blocked this caller.

**spam_reports_count**             - How many times this caller has been reported as spam.

**country_code_risk_score**        - Risk score assigned to the caller's country/region code, based on historical fraud rates from that region.

**night_call_ratio**               - Proportion of this caller's calls made during night hours. Unusual night-calling patterns can be a fraud signal.

**sequential_dialing_score**       - Measures how "sequential" or automated the calling pattern looks (e.g. dialing numbers in order), typical of robocalls/scripted fraud.

**graph_degree**                   - Number of connections (distinct numbers/people) this caller has in the call network.

**previous_fraud_associations**    - Count of past connections to numbers/accounts previously flagged as fraudulent.

**reputation_score**               - Overall trust/reputation score for the caller (higher = more trustworthy).

**fraud_label**                    - Target variable: 1 = fraudulent call, 0 = legitimate call.
