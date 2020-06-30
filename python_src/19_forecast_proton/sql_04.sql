select id, 
	time_tag, datetime('now') as now , 
	round( ( julianday( datetime( '1998-02-04 00:01') ) - julianday( time_tag ) )*86400.0 )
from xray limit 100 