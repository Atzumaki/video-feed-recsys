-- Point-in-time-correct training dataset (ClickHouse SQL dialect).
-- Every feature is computed strictly before the impression timestamp.
WITH impressions AS (
    SELECT
        user_id,
        video_id,
        category_id,
        impression_ts,
        watched_ms,
        video_duration_ms,
        purchased_within_24h,
        ad_cpm
    FROM mart.video_impressions
    WHERE impression_date BETWEEN {start_date:Date} AND {end_date:Date}
),
user_history AS (
    SELECT
        current.user_id,
        current.video_id,
        current.impression_ts,
        countIf(history.watched_ms >= 3000) AS watches_7d,
        avgIf(history.watched_ms / history.video_duration_ms, history.video_duration_ms > 0)
            AS mean_completion_7d,
        groupArray(20)(history.category_id) AS recent_categories
    FROM impressions AS current
    LEFT JOIN mart.video_impressions AS history
        ON current.user_id = history.user_id
        AND history.impression_ts < current.impression_ts
        AND history.impression_ts >= current.impression_ts - INTERVAL 7 DAY
    GROUP BY current.user_id, current.video_id, current.impression_ts
)
SELECT
    i.user_id,
    i.video_id,
    i.category_id,
    i.impression_ts,
    h.watches_7d,
    h.mean_completion_7d,
    h.recent_categories,
    i.ad_cpm,
    toUInt8(i.watched_ms >= 3000) AS target_watch,
    toUInt8(i.purchased_within_24h) AS target_purchase
FROM impressions AS i
INNER JOIN user_history AS h USING (user_id, video_id, impression_ts);

