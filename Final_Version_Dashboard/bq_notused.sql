SELECT
          table_name, max(audit_tables.createTime) AS last_query_date, table_schema
FROM
      region-us.INFORMATION_SCHEMA.TABLES AS tables
LEFT JOIN
      bq_audits.bq_query_audit_tables AS audit_tables
      ON tables.table_catalog= audit_tables.projectId AND tables.table_schema= audit_tables.datasetId AND tables.table_name= audit_tables.tableId
      AND CASE tables.table_type WHEN 'BASE TABLE' THEN audit_tables.table_type ='table' WHEN 'VIEW' THEN  audit_tables.table_type ='view' END
GROUP BY table_name, table_schema
 