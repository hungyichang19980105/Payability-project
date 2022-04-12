SELECT
      projectId, datasetId AS table_schema, createTime As Query_Time, tableId, statementType, audit_tables.table_type AS table_type
FROM
      bq_audits.bq_query_audit_tables AS audit_tables
LEFT JOIN
         region-us.INFORMATION_SCHEMA.TABLES AS tables
            ON tables.table_catalog= audit_tables.projectId AND tables.table_schema= audit_tables.datasetId AND tables.table_name= audit_tables.tableId
            AND CASE tables.table_type WHEN 'BASE TABLE' THEN audit_tables.table_type ='table' WHEN 'VIEW' THEN  audit_tables.table_type ='view' END
GROUP BY projectId, table_schema, createTime, tableId, statementType, table_type