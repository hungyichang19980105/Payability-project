SELECT
          isError AS ErrorTpye,
          totalBilledTerabytes AS BilledTB,
          estimatedCostUsd AS Cost,
          runtimeSecs AS runningTime,
          principalEmail AS User,
          createTime
FROM bq_audits.bq_query_audit
GROUP BY User, ErrorTpye, BilledTB, Cost, runningTime, createTime