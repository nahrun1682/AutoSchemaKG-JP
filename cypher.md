CALL apoc.import.graphml("file://Dulce.graphml", {
    readLabels: true,
    storeNodeIds: true
})


// エンティティ

MATCH (n)

WHERE n.type = 'entity'

SET n:Entity;


// イベント

MATCH (n)

WHERE n.type = 'event'

SET n:Event;


// パッセージノードがあれば

MATCH (n)

WHERE n.type = 'passage'

SET n:Passage;


// Concept ノードもあるなら

MATCH (n)

WHERE n.type = 'concept'

SET n:Concept;

# リレーション格上げ
MATCH (a)-[r:RELATED]->(b)
WITH a, b, r,
     CASE r.type
       WHEN "Relation" THEN "RELATION"
       WHEN "Source"  THEN "SOURCE"
       WHEN "Concept" THEN "CONCEPT"
     END AS relType
CALL apoc.create.relationship(
  a,
  relType,
  apoc.map.removeKey(properties(r), 'type'),
  b
) YIELD rel
DELETE r;

// Entity ラベルで最も接続数が多いノードTop10（最新版）
MATCH (e:Entity)
WITH e, COUNT { (e)--() } as connections
ORDER BY connections DESC
LIMIT 10
RETURN 
  e.id as エンティティ名,
  connections as 接続数,
  labels(e) as ラベル

