SELECT
table_catalog, table_schema, table_name, tables.table_type AS table_type1, creation_time AS table_creation_time, COUNT(*) as run_cnt
FROM
	region-us.INFORMATION_SCHEMA.TABLES AS tables
LEFT JOIN
          bq_audits.bq_query_audit_tables AS audit_tables
              ON tables.table_catalog= audit_tables.projectId AND tables.table_schema= audit_tables.datasetId AND tables.table_name= audit_tables.tableId
              AND CASE tables.table_type WHEN 'BASE TABLE' THEN audit_tables.table_type ='table' WHEN 'VIEW' THEN  audit_tables.table_type ='view' END
WHERE DATE(createTime) >= CURRENT_DATE() - 90
GROUP BY table_catalog, table_schema, table_name, table_type1, table_creation_time
ORDER BY run_cnt DESC