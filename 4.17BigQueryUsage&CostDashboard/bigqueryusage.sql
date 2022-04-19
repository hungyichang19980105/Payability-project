SELECT
          table_catalog, table_schema, 
          table_name, tables.table_type AS table_type1, 
          creation_time AS table_creation_time,
          audit_tables.createTime As run_time,
          audit_tables.principalEmail,
          audit_tables.statementType,
          audit_tables.jobId,
          audit_tables.table_type,
          max(audit_tables.createTime) AS last_query_date,
          Query_Table.isError,
          Query_Table.totalBilledGigabytes,
          Query_Table.totalBilledTerabytes,
          Query_Table.estimatedCostUsd,
          Query_Table.runtimeSecs
FROM
      region-us.INFORMATION_SCHEMA.TABLES AS tables
LEFT JOIN
      bq_audits.bq_query_audit_tables AS audit_tables
      ON tables.table_catalog= audit_tables.projectId AND tables.table_schema= audit_tables.datasetId AND tables.table_name= audit_tables.tableId
      AND CASE tables.table_type WHEN 'BASE TABLE' THEN audit_tables.table_type ='table' WHEN 'VIEW' THEN  audit_tables.table_type ='view' END
LEFT JOIN
      bq_audits.bq_query_audit AS Query_Table
      ON audit_tables.jobID = Query_Table.jobID
GROUP BY   table_catalog, table_schema, table_name, table_type1, table_creation_time, run_time, audit_tables.principalEmail, audit_tables.statementType, audit_tables.jobId, audit_tables.table_type, Query_Table.isError, Query_Table.totalBilledGigabytes, Query_Table.totalBilledTerabytes, Query_Table.estimatedCostUsd, Query_Table.runtimeSecs