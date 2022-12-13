-- Uncomment below lines and the last line to test in BQ CLI with example parameters:
-- bq query \
--     --use_legacy_sql=false \
--     --parameter=start_date:STRING:"20210101" \
--     --parameter=end_date:STRING:"20210102" \
--     --parameter=sample_size:INT64:10 \
-- '
SELECT * FROM (
  SELECT
    s.full_visitor_id,
    s.visit_start_time,
    s.date,
    s.device_category,
    s.is_mobile,
    s.operating_system,
    s.browser,
    s.country,
    s.city,
    s.traffic_source,
    s.traffic_medium,
    s.traffic_campaign,
    s.is_first_visit,
    s.product_pages_viewed,
    u.total_hits,
    u.total_pageviews,
    u_visits.total_visits,
    u_engagement.total_time_on_site,
    s.added_to_cart
  FROM (
        -- sessions
        SELECT
          full_visitor_id,
          ga_session_id,
          MAX(visit_start_time) AS visit_start_time,
          date, device_category, is_mobile, operating_system, browser, 
          country, city, traffic_source, traffic_medium, traffic_campaign,
          MAX(is_first_visit) AS is_first_visit, 
          MAX(product_pages_viewed) AS product_pages_viewed, 
          MAX(added_to_cart) AS added_to_cart
        FROM (
          -- raw events
          SELECT 
            user_pseudo_id AS full_visitor_id,
            ep.value.int_value AS ga_session_id,
            CASE event_name WHEN "session_start" THEN event_timestamp  ELSE NULL END AS visit_start_time,
            event_date AS date,
            device.category AS device_category,
            CASE device.category WHEN "mobile" THEN 1  ELSE 0 END AS is_mobile,
            device.operating_system AS operating_system,
            device.web_info.browser AS browser,
            geo.country AS country,
            geo.city AS city,
            traffic_source.source AS traffic_source,
            traffic_source.medium AS traffic_medium,
            CASE traffic_source.medium  WHEN "cpc" THEN traffic_source.name  ELSE NULL END AS traffic_campaign,
            CASE event_name WHEN "first_visit" THEN 1  ELSE 0 END AS is_first_visit,
            CASE event_name WHEN "view_item" THEN 1  ELSE 0 END AS product_pages_viewed,
            CASE event_name WHEN "add_to_cart" THEN 1  ELSE 0 END AS added_to_cart
          FROM
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` e
          CROSS JOIN
            UNNEST(e.event_params) ep 
          WHERE
            ep.key = "ga_session_id"
            AND _TABLE_SUFFIX BETWEEN @start_date AND @end_date
        ) events
        GROUP BY
          full_visitor_id, ga_session_id, date, device_category, is_mobile, 
          operating_system, browser, country, city, traffic_source, traffic_medium, traffic_campaign
        ) s
  LEFT JOIN (
      -- total_hits, total_pageviews
      SELECT
        user_pseudo_id AS full_visitor_id, 
        COUNT(user_pseudo_id) AS total_hits,
        COUNTIF(event_name = "page_view") AS total_pageviews,
      FROM
        `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` e
      WHERE
        _TABLE_SUFFIX BETWEEN @start_date AND @end_date
      GROUP BY
        user_pseudo_id
  ) u USING(full_visitor_id)
  LEFT JOIN (
      -- total_visits
      -- count unique `ga_session_id`
      SELECT
        user_pseudo_id AS full_visitor_id, 
        COUNT(DISTINCT ep.value.int_value) AS total_visits
      FROM
        `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` e
      CROSS JOIN
        UNNEST(e.event_params) ep
      WHERE
        ep.key = "ga_session_id"
        AND _TABLE_SUFFIX BETWEEN @start_date AND @end_date     
      GROUP BY
        user_pseudo_id
  ) u_visits USING(full_visitor_id)
  LEFT JOIN (
      -- total_time_on_site (divide by 1000 to get seconds)
      SELECT
        user_pseudo_id AS full_visitor_id, 
        MAX(engagement_time_msec) AS total_time_on_site
      FROM (
        SELECT 
          user_pseudo_id, 
          value.int_value AS engagement_time_msec
        FROM 
          `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` e
        CROSS JOIN
          UNNEST(e.event_params) ep
        WHERE
          key = "engagement_time_msec"
          AND _TABLE_SUFFIX BETWEEN @start_date AND @end_date
      ) s
      GROUP BY user_pseudo_id
  ) u_engagement USING(full_visitor_id)
)
ORDER BY RAND()
LIMIT @sample_size
-- '