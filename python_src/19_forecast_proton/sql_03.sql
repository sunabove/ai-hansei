select time_tag, 
xs - lag( xs , 1 ) over( where xs is not null order by time_tag ) as dxs 
from xray 
where phase = 'train' 
order by phase, time_tag, id
limit 100