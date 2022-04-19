SELECT
          table_name, max(audit_tables.createTime) AS last_query_date
FROM
      region-us.INFORMATION_SCHEMA.TABLES AS tables
LEFT JOIN
      bq_audits.bq_query_audit_tables AS audit_tables
      ON tables.table_catalog= audit_tables.projectId AND tables.table_schema= audit_tables.datasetId AND tables.table_name= audit_tables.tableId
      AND CASE tables.table_type WHEN 'BASE TABLE' THEN audit_tables.table_type ='table' WHEN 'VIEW' THEN  audit_tables.table_type ='view' END
LEFT JOIN
      bq_audits.bq_query_audit AS Query_Table
      ON audit_tables.jobID = Query_Table.jobID
GROUP BY table_name