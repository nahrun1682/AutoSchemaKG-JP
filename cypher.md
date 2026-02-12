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