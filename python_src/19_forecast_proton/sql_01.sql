-- select id, time_tag, strftime('%d-%m-%Y %H:%M:%f', time_tag) from proton ;

-- select id, time_tag, strftime('%d-%m-%Y %H:%M:%S', date( replace( replace( replace( time_tag, 'T', ' ' ), 'Z', ' ' ), ':00.000', '' ), 'YYYY-MM-DD hh:mi:ss' ) ) from epm ; 

-- update epm set time_tag = replace( replace( replace( time_tag, 'T', ' ' ), 'Z', ' ' ), ':00.000', '' ) 

update swe set time_tag = replace( replace( replace( time_tag, 'T', ' ' ), 'Z', ' ' ), ':00.000', '' ) 