---
name: table
intents: tabla
formats: markdown_table
response_format: markdown_table
max_tokens: 4000
post_node: table_node
---

El usuario pide una tabla. La evidencia debe contener TODAS las filas de la
tabla solicitada; si sospechas que faltan (rangos interrumpidos, numeración
salteada), re-consulta usando la referencia literal (p. ej. "tabla 220.55").
Incluye en tu respuesta los datos tabulares COMPLETOS, sin resumir ni omitir
filas, conservando los valores tal como aparecen (rangos, fórmulas, unidades).
El render visual de la tabla lo hace otro nodo: tu trabajo es que no falte
ninguna fila.
