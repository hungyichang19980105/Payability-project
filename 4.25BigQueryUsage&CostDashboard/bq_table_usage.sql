SELECT
          table_catalog, table_schema,
          table_name,
          creation_time AS table_creation_time,
          audit_tables.createTime AS run_time,
          audit_tables.statementType AS StatementType,
          audit_tables.table_type As TableType
FROM
      region-us.INFORMATION_SCHEMA.TABLES AS tables
LEFT JOIN
      bq_audits.bq_query_audit_tables AS audit_tables
      ON tables.table_catalog= audit_tables.projectId AND tables.table_schema= audit_tables.datasetId AND tables.table_name= audit_tables.tableId
      AND CASE tables.table_type WHEN 'BASE TABLE' THEN audit_tables.table_type ='table' WHEN 'VIEW' THEN  audit_tables.table_type ='view' END
GROUP BY   table_catalog, table_schema, table_name, table_creation_time, run_time, StatementType, TableType
 