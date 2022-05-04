SELECT
          Cast(isError as string) AS ErrorType,
          totalBilledTerabytes AS BilledTB,
          estimatedCostUsd AS Cost,
          runtimeSecs AS runningTime,
          principalEmail AS User,
          createTime
FROM bq_audits.bq_query_audit
GROUP BY User, ErrorType, BilledTB, Cost, runningTime, createTime