-- osm2pgsql flex style: load the elements osm.SELECT picks into PostGIS, with raw
-- tags, way node ids and relation members, so the database mirrors osm.read_pbf.
--   osm2pgsql -d osm -O flex -S osm_flex.lua outputs/osm/pbf/bayern-latest.osm.pbf
-- The tag rules copy osm.SELECT (ROAD_WIDTH, RAIL_WIDTH | RAIL_INACTIVE); keep both in sync.

local function set(list)
    local s = {}
    for _, v in ipairs(list) do s[v] = true end
    return s
end

local HIGHWAY = set{'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'motorway_link',
                    'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link', 'residential',
                    'unclassified', 'living_street', 'service', 'track'}
local LEISURE = set{'pitch', 'track', 'swimming_pool', 'marina'}
local AEROWAY = set{'helipad', 'aerodrome'}
local RAILWAY = set{'rail', 'light_rail', 'narrow_gauge', 'abandoned', 'disused'}

-- Postgres array literal of the cache kinds the object belongs to, or nil.
local function kinds(t, tags)
    local k = {}
    if (t == 'w' and HIGHWAY[tags.highway]) or LEISURE[tags.leisure] or tags.sport == 'athletics'
            or tags.man_made == 'storage_tank' or (t == 'w' and tags.bridge == 'yes')
            or AEROWAY[tags.aeroway] then
        k[#k + 1] = 'raw'
    end
    if t == 'w' and RAILWAY[tags.railway] then k[#k + 1] = 'rail' end
    if t ~= 'n' and (tags.landuse == 'forest' or tags['natural'] == 'wood') then k[#k + 1] = 'forest' end
    if t ~= 'n' and tags.building then k[#k + 1] = 'building' end
    if #k > 0 then return '{' .. table.concat(k, ',') .. '}' end
end

local nodes = osm2pgsql.define_node_table('osm_nodes', {
    { column = 'kinds', sql_type = 'text[]', not_null = true },
    { column = 'tags', type = 'jsonb' },
    { column = 'geom', type = 'point', projection = 4326, not_null = true },
})

local ways = osm2pgsql.define_way_table('osm_ways', {
    { column = 'kinds', sql_type = 'text[]', not_null = true },
    { column = 'tags', type = 'jsonb' },
    { column = 'nodes', sql_type = 'int8[]', not_null = true },
    { column = 'geom', type = 'linestring', projection = 4326 },  -- full way, closed ones too
})

local relations = osm2pgsql.define_relation_table('osm_relations', {
    { column = 'kinds', sql_type = 'text[]', not_null = true },
    { column = 'tags', type = 'jsonb' },
    { column = 'members', type = 'jsonb' },
    { column = 'geom', type = 'multipolygon', projection = 4326 },  -- assembled; NULL if invalid
})

function osm2pgsql.process_node(object)
    local k = kinds('n', object.tags)
    if k then nodes:insert{ kinds = k, tags = object.tags, geom = object:as_point() } end
end

function osm2pgsql.process_way(object)
    local k = kinds('w', object.tags)
    if k then
        ways:insert{ kinds = k, tags = object.tags, nodes = '{' .. table.concat(object.nodes, ',') .. '}',
                     geom = object:as_linestring() }
    end
end

function osm2pgsql.process_relation(object)
    local k = kinds('r', object.tags)
    if k then
        relations:insert{ kinds = k, tags = object.tags, members = object.members,
                          geom = object:as_multipolygon() }
    end
end
