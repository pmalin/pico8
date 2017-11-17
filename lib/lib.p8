pico-8 cartridge // http://www.pico-8.com
version 14
__lua__

dbg = 0
perf = true

perf_counters = {}

----------------------
-- system
----------------------

-- misc

function ceil(num)
 return flr(num+0x0.ffff)
end

function min3(a,b,c)
 return min(a, min(b,c))
end

function max3(a,b,c)
 return max(a, max(b,c))
end

function sys_get_framrate()
 if _update60 then 
  return 60
 end
  return 30
end

sys_time = { t=0, dt=1/60, fr=60 }

function sys_time_tick()
 sys_time.fr = sys_get_framrate()
 sys_time.dt=1/sys_time.fr
 sys_time.t+=sys_time.dt
end

-- perf

function perf_counter( key )
 if (not perf) return
 perf_counters[key] = { name = key, total = 0, avg = 0 }
end

function perf_reset()
 if (not perf) return
 for k,c in pairs(perf_counters) do
  c.avg = c.avg * 0.95 + c.total * 0.05
  c.total = 0
  c.count = 0
 end
end

function perf_timer()
 return 1000 * stat(1) / 60
end

function perf_begin( key )
 if (not perf) return
 c = perf_counters[key]
 if c then 
  c.start = perf_timer()
 end
end

function perf_end( key )
 if (not perf) return
 c = perf_counters[key]
 if c then 
  c.total += perf_timer() - c.start
  c.count += 1
  c.start = nil 
 end
end

function perf_draw_timers()
 if (not perf) return
 cursor(0,0)
 color(7)
 for k,c in pairs(perf_counters) do
  print(c.name .. "[" .. c.count .. "]=" .. (c.total) .. ",".. c.avg)
 end
end

function perf_hud()
 clip()
 local cpu=flr(stat(1)*100)
 local fps=sys_time.fr/ceil(stat(1))
 local mem=flr(stat(0))
 local perf=
  cpu .. "% cpu @ " ..
  fps ..  " fps " ..
  mem .. " mb"
 print(perf,0,122,0)
 print(perf,0,121,fps==sys_time.fr and 7 or 8)

 perf_draw_timers()
end

-- vector2

function v2(x,y)
 return {x,y}
end

function v2_tostring(a)
 return "(" .. a[1] .. "," .. a[2] .. ")"
end

function v2_neg(a)
 return v2(-a[1], -a[2])
end

function v2_rcp(a)
 return v2(1/a[1], 1/a[2])
end

function v2_add(a,b)
 return v2(a[1]+b[1], a[2]+b[2])
end

function v2_add_s(a,b)
 return v2(a[1]+b, a[2]+b)
end

function v2_sub(a,b)
 return v2(a[1]-b[1], a[2]-b[2])
end

function v2_mul(a, b)
	return v2( a[1] * b[1], a[2] * b[2] )
end

function v2_mul_s(a, b)
    return v2( a[1] * b, a[2] * b )
end

function v2_cross(a,b)
	return a[1]*b[2]-a[2]*b[1]
end

-- vector3 

function v3(x,y,z)
	return {x,y,z}
end

function v3_tostring(a)
	return "(" .. a[1] .. "," .. a[2] .. "," .. a[3] .. ")"
end

function v3_neg(a)
	return v3(-a[1], -a[2], -a[3])
end

function v3_rcp(a)
 return v3(1/a[1], 1/a[2], 1/a[3])
end

function v3_add(a,b)
	return v3(a[1]+b[1], a[2]+b[2], a[3]+b[3])
end

function v3_add_s(a,b)
	return v3(a[1]+b, a[2]+b, a[3]+b)
end

function v3_sub(a,b)
	return v3(a[1]-b[1], a[2]-b[2], a[3]-b[3])
end

function v3_sub_s(a,b)
	return v3(a[1]-b, a[2]-b, a[3]-b)
end

function v3_mul(a,b)
	return v3(a[1]*b[1],a[2]*b[2],a[3]*b[3])
end

function v3_mul_s(a, b)
	return v3( a[1]*b, a[2]*b, a[3]*b )
end

function v3_dot(a, b)
	return a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
end

function v3_cross(a, b)
	return v3( a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1] )
end

function v3_min(a,b)
	return v3( min(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3]) )
end

function v3_max(a,b)
	return v3( max(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3]) )
end

function v3_length(a)
	return sqrt( v3_dot(a,a) )
end	

function v3_normalize(a)
	return v3_mul_s( a, 1 / v3_length(a) )
end

function v3_rot_x(a, t)
	local s = sin(t)
	local c = cos(t)
	return v3( a[1], a[2] * c + a[3] * s, a[2] * -s + a[3] * c )	
end

function v3_rot_y(a, t)
	local s = sin(t)
	local c = cos(t)
	return v3( a[1] * c + a[3] * s, a[2], a[1] * -s + a[3] * c )	
end

-- plane
function pl(n,d)
 return { n[1], n[2], n[3], d }
end

function pl_abc(a,b,c)
 local ba = v3_sub(a,b)
 local bc = v3_sub(c,b)
 local n = v3_normalize( v3_cross(ba, bc) )
 return pl( n, -v3_dot(n, a) )
end

function pl_dist(p, v)
 return v3_dot(p,v) + p[4]
end

-- matrix

function m3(x,y,z)
	return {{x[1],x[2],x[3]},{y[1],y[2],y[3]},{z[1],z[2],z[3]}}
end	

function m3_id()
	return {{1,0,0},{0,1,0},{0,0,1}}
end	

function m3_get_ax(m)
 return v3(m[1][1], m[2][1], m[3][1])
end
function m3_get_ay(m)
 return v3(m[1][2], m[2][2], m[3][2])
end

function m3_get_az(m)
 return v3(m[1][3], m[2][3], m[3][3])
end

function m3_rot_x(t)
	local s = sin(t)
	local c = cos(t)
	return m3(v3(1, 0, 0), v3(0, c,  s), v3(0,  -s,  c))
end

function m3_rot_y(t)
	local s = sin(t)
	local c = cos(t)
	return m3(v3(c, 0, -s), v3(0,  1,  0), v3(s,  0,  c))
end

function m3_rot_z(t)
	local s = sin(t)
	local c = cos(t)
	return m3(v3(c, -s, 0), v3(s,  c, 0), v3(0,  0, 1))
end

function m3_mul(m1, m2)
	local r = {}
	for i=1,3 do
		r[i] = {}
		for j=1,3 do
			local v = m1[i][1] * m2[1][j]
			for k=2,3 do
				v += m1[i][k] * m2[k][j]
			end
			r[i][j] = v
		end
	end
	return r
end

function m3_trans(m)
	local r = {}
	for i=1,3 do
		r[i] = {}
		for j=1,3 do
			r[i][j] = m[j][i]
		end
	end
	return r
end

function v3_mul_m3(v, m)
	return v3( v[1] * m[1][1] + v[2] * m[1][2] + v[3] * m[1][3],
	 v[1] * m[2][1] + v[2] * m[2][2] + v[3] * m[2][3],
	 v[1] * m[3][1] + v[2] * m[3][2] + v[3] * m[3][3] )
end

-- rot-trans

-- r11,r12,r13, r21,r22,r23, r31,r32,r33, tx, ty, tz = rt_unpack(rt)
function rt_unpack(rt)
 local r = rt.r
 local r1 = r[1]
 local r2 = r[2]
 local r3 = r[3] 
 local t = rt.t
 return r1[1],r1[2],r1[3], r2[1],r2[2],r2[3], r3[1],r3[2],r3[3], t[1],t[2],t[3]
end

function rt_apply(v, rt)
 local r = rt.r
 local r1 = r[1]
 local r2 = r[2]
 local r3 = r[3] 
 local t = rt.t
 return v3( t[1] + v[1] * r1[1] + v[2] * r1[2] + v[3] * r1[3],
  t[2] + v[1] * r2[1] + v[2] * r2[2] + v[3] * r2[3],
  t[3] + v[1] * r3[1] + v[2] * r3[2] + v[3] * r3[3] )
end

function rt_apply_unpacked(vx,vy,vz, r11,r12,r13, r21,r22,r23, r31,r32,r33, tx, ty, tz )
 return 
  tx + vx * r11 + vy * r12 + vz * r13,
  ty + vx * r21 + vy * r22 + vz * r23,
  tz + vx * r31 + vy * r32 + vz * r33
end

function rt_mul(rta, rtb)
 local rt = {}
 rt.r = m3_mul(rtb.r, rta.r)
 rt.t = v3_add(rtb.t, v3_mul_m3(rta.t, rtb.r))
 return rt
end

function rt_inv(rt)
	local r = m3_trans(rt.r)
	return { r = r, t = v3_mul_m3( v3_neg(rt.t), r ) }
end	

-- sort
-- casualeffects heap sort:
-- https://github.com/morgan3d/misc/blob/master/p8sort/sort.p8

function ce_heap_sort(data)
 local n = #data

 if (n==0) return

 -- form a max heap
 for i = flr(n / 2) + 1, 1, -1 do
  -- m is the index of the max child
  local parent, value, m = i, data[i], i + i
  local key = value.key 
  
  while m <= n do
   -- find the max child
   if ((m < n) and (data[m + 1].key > data[m].key)) m += 1
   local mval = data[m]
   if (key > mval.key) break
   data[parent] = mval
   parent = m
   m += m
  end
  data[parent] = value
 end 

 -- read out the values,
 -- restoring the heap property
 -- after each step
 for i = n, 2, -1 do
  -- swap root with last
  local value = data[i]
  data[i], data[1] = data[1], value

  -- restore the heap
  local parent, terminate, m = 1, i - 1, 2
  local key = value.key 
  
  while m <= terminate do
   local mval = data[m]
   local mkey = mval.key
   if (m < terminate) and (data[m + 1].key > mkey) then
    m += 1
    mval = data[m]
    mkey = mval.key
   end
   if (key > mkey) break
   data[parent] = mval
   parent = m
   m += m
  end  
  
  data[parent] = value
 end
end


-- view state

function vs_cull_sphere( o, r )
 local pc = #vs.frustum.planes
 for pi=1,pc do
  local p=vs.frustum.planes[pi]
  if (pl_dist(p,o) < -r) then
   --gfx_3d_sphere_outline( o, r * 2, 8 )
   return false
  end
 end 
 --gfx_3d_sphere_outline( o, r, 11 )
 return true
end

vs = {}

function vs_frustum_setup()
 vs.frustum = {}
 vs.frustum.planes = {}

 vcam = vs.cam_to_world.t
 vptl = v3(vs.vp.tl[1], vs.vp.tl[2], vs.pdist)
 vptr = v3(vs.vp.br[1], vs.vp.tl[2], vs.pdist)
 vpbr = v3(vs.vp.br[1], vs.vp.br[2], vs.pdist)
 vpbl = v3(vs.vp.tl[1], vs.vp.br[2], vs.pdist)

 vptl = rt_apply( vptl, vs.cam_to_world )
 vptr = rt_apply( vptr, vs.cam_to_world )
 vpbr = rt_apply( vpbr, vs.cam_to_world )
 vpbl = rt_apply( vpbl, vs.cam_to_world )

 add( vs.frustum.planes, pl_abc(vcam, vpbl, vptl))
 add( vs.frustum.planes, pl_abc(vcam, vptl, vptr))
 add( vs.frustum.planes, pl_abc(vcam, vptr, vpbr))
 add( vs.frustum.planes, pl_abc(vcam, vpbr, vpbl))
end

function vs_view_setup( cam_to_world, vp, pdist, win, sc )
 vs.near = 0.1
 vs.vp = vp
 vs.pdist = pdist
 vs.win = win 
 if (sc == nil) vs.sc = win else vs.sc = sc

 vs.sct = vs.sc.tl[2]
 vs.scb = vs.sc.br[2] - 1
 vs.scl = vs.sc.tl[1]
 vs.scr = vs.sc.br[1] - 1

 vs.vp.size = v2_sub( vs.vp.br, vs.vp.tl )  
 vs.vp.rcp_size = v2_rcp( vs.vp.size ) 

 vs.win.size = v2_sub( vs.win.br, vs.win.tl )

 vs.st_scale = v2_mul( vs.vp.rcp_size, vs.win.size )
 vs.st_offset = v2_mul( v2_sub( vs.win.tl, vs.vp.tl ), vs.st_scale )

 vs.cam_to_world = cam_to_world
 vs.world_to_cam = rt_inv( cam_to_world )
 vs.cam_z = m3_get_az( cam_to_world.r )

 vs_frustum_setup()
end

function vs_set_obj_mat( obj_to_world )
   vs.obj_to_world = obj_to_world
   vs.obj_to_cam = rt_mul( obj_to_world, vs.world_to_cam )

   local world_to_shadow = 
   {
    r = { v3(1,0.3,0), v3(0,0,0), v3(0,0,1) },
    t = v3(0,0,0)
   }
   vs.obj_to_cam_shadow = rt_mul( rt_mul( obj_to_world, world_to_shadow ), vs.world_to_cam )
   vs.world_to_obj_rot = m3_trans( obj_to_world.r )
end


-- sx,sy,ox,oy
function vs_unpack_view_to_screen()
 return vs.st_scale[1] * vs.pdist, vs.st_scale[2] * vs.pdist, vs.st_offset[1], vs.st_offset[2]
end

function vs_view_to_screen_unpacked( x,y,z, sx,sy,ox,oy )
 local w = 1. / z
 return { x * w * sx + ox, y * w * sy + oy, z, w }
end


function vs_view_to_screen( v )
 -- project onto screen at distance vs.pdist in view space
 local w = 1. / v[3]
 local ps = v2_add(v2_mul(v2_mul_s( v, vs.pdist * w ), vs.st_scale), vs.st_offset)
 ps.w = w
 return {ps[1], ps[2], v[3], w }
end

function vs_screen_to_viewport( s )
 return v2_mul( v2_sub( s, vs.st_offset ), v2_rcp(vs.st_scale) )
end

-- gfx

drawlist = {}

function dl_reset()
 drawlist = {}
end

function dl_draw()
 sortlist = {}

 --for k,v in pairs(drawlist) do
 for i=1,#drawlist do
  local v = drawlist[i]
  add(sortlist, {key=-v.key,dl_key=i})
 end
 
 ce_heap_sort(sortlist)

 local slc = #sortlist
 for i=1,slc do
  local p = sortlist[i]
  local item = drawlist[p.dl_key]
  item.fn( item.value )
 end
end


function dl_tri(v)
   fillp(v.fp)
   local a=v.a
   local b=v.b
   local c=v.c
   trifill( a[1],a[2], b[1],b[2], c[1],c[2], v.col )
end   

function dl_line(v)
   fillp()
   line(v.a[1], v.a[2], v.b[1], v.b[2], v.col)
end   



function gfx_line( a, b, col )
  line(a[1], a[2], b[1], b[2], col)
end

function gfx_point( a, col )
  pset(a[1], a[2], col)
end


function gfx_3d_line( a, b, col )
  local ac = rt_apply( a, vs.world_to_cam )
  local bc = rt_apply( b, vs.world_to_cam )

  if ac[3] < vs.near then
   if bc[3] < vs.near then
    return
   else
    -- clip a
    atob = v3_sub(bc, ac)
    t = (vs.near - ac[3]) / atob[3]
    ac = v3_add( ac, v3_mul_s( atob, t ) )
   end
  elseif bc[3] < vs.near then
    -- clip b
    atob = v3_sub(bc, ac)
    t = (vs.near - ac[3]) / atob[3]
    bc = v3_add( ac, v3_mul_s( atob, t ) )
  end

  local as = vs_view_to_screen( ac )
  local bs = vs_view_to_screen( bc )
  
  gfx_line(as, bs, col)
end 

function gfx_3d_grid( col )
 if (not vs_cull_sphere( v3(0,0,0), sqrt(50) )) then return end

 local s = 4
 for i=-s, s do
  gfx_3d_line( v3(i, 0, -s), v3(i, 0, s), col )
  gfx_3d_line( v3(-s, 0, i), v3(s, 0, i), col )
 end
end

function get_projected_size( r, d )
 local f = d * d - r * r
 if ( f > 0 ) return d * r / sqrt( f ) else return -1 
end

function gfx_3d_sphere_outline( c, r, col )
  --if (not vs_cull_sphere( c, r )) then return end
  c = rt_apply( c, vs.world_to_cam )
  local cp = vs_view_to_screen( c )
  local d = v3_length(c)

  local p = get_projected_size( r, d )
   
  local c2 = v3(c[1], c[2], c[3])
  c2[1] += p 
  local cp2 = vs_view_to_screen( c2 )

  if ( p > 0 ) circ( cp[1], cp[2], cp2[1] - cp[1], col )
end 

function gfx_3d_sprite( c, w, h, sx, sy, sw, sh )
  c = rt_apply( c, vs.world_to_cam )
  if (c[3] < vs.near ) return

  d = v3_length(c)

  px = get_projected_size( w, d )
  py = get_projected_size( h, d )
   
  c1 = v3_sub( c, v3( px, py, 0) )
  c2 = v3_add( c, v3( px, py, 0) )
  cp1 = vs_view_to_screen( c1 )
  cp2 = vs_view_to_screen( c2 )

  sspr( sx, sy, sw, sh, cp1[1], cp1[2], cp2[1] - cp1[1], cp2[2] - cp1[2] )
end 
  
function gfx_3d_print(p, str, col)
  pw = rt_apply( p, vs.world_to_cam )
  pv = vs_view_to_screen( pw )
  if (pv[3] > vs.near) print(str, pv[1], pv[2], col )
end


function tri( a, b, c, col )
	line(a[1], a[2], b[1], b[2], col)
	line(b[1], b[2], c[1], c[2], col)
	line(c[1], c[2], a[1], a[2], col)
end

function sort_y_3( ax,ay, bx,by, cx,cy )
 if ay > by then
  ax,ay,bx,by = bx,by,ax,ay
 end

 if by > cy then
  bx,by,cx,cy = cx,cy,bx,by
  if ay > by then
   ax,ay,bx,by = bx,by,ax,ay
  end  
 end
 return ax,ay, bx,by, cx,cy
end

function trifill( ax,ay, bx,by, cx,cy, col )

	-- sort vertices
 ax,ay,bx,by,cx,cy = sort_y_3( ax,ay, bx,by, cx,cy )

 -- ceil and clip y
 ayc = mid( ceil(ay), vs.sct, vs.scb )
 byc = mid( ceil(by), vs.sct, vs.scb )
 cyc = mid( ceil(cy), vs.sct, vs.scb )

 -- init edges
	dabx_dy = (bx - ax) / (by - ay)
	dacx_dy = (cx - ax) / (cy - ay)
 dbcx_dy = (cx - bx) / (cy - by)

	ab_x = ax + (ayc - ay) * dabx_dy
	ac_x = ax + (ayc - ay) * dacx_dy
 bc_x = bx + (byc - by) * dbcx_dy

 byc-=1

	for py=ayc,byc do
		rectfill( ab_x, py, ac_x, py, col )
		ab_x += dabx_dy
		ac_x += dacx_dy
	end

 byc+=1
 cyc-=1

	for py=byc,cyc do
  rectfill( bc_x, py, ac_x, py, col )
		bc_x += dbcx_dy
		ac_x += dacx_dy
	end
end


-- fillp dither patterns

dither_patterns = {
0b0000000000000000,
0b0000000000100000,
0b1000000000100000,
0b1001000000100000,
0b1001000000100100,
0b1001010000100100,
0b1011010000100100,
0b1001010010100100,
0b1011010010100100,
0b1011010011100100,
0b1011011011100100,
0b1011011011101100,
0b1011011011111100,
0b1011011011111101,
0b1111011011111101,
0b1111011111111101,
0b1111011111111111
}

function gfx_dither_calc( grad, s )
   local gc = #grad
   local g = s * (gc-1)
   local g_i = flr(g)   
   local g_f = g - g_i

   local g_i1 = mid( g_i + 1, 1, gc )
   local g_i2 = mid( g_i + 2, 1, gc )

   local c1 = grad[g_i1]
   local c2 = grad[g_i2]
   local col = c1 + c2 * 16 

   local dc = #dither_patterns
   local fp = dither_patterns[ mid( 1 + flr( g_f * (dc-1) ), 1, dc ) ]

   return {c = col, f = fp }
end

gradients = { 
 {0x0, 0x2, 0x4, 0x9, 0xa, 0x7},  --yellow
 {0x0, 0x1, 0x3, 0xb, 0x7}, -- green
 {0x0, 0x1, 0xc, 0x7}, -- blue
 {0x0, 0x2, 0x8, 0xe, 0x7}  -- red
}

dither_tables = {}

function init_dither()
 dither_tables = {}
 for grad in all(gradients) do
  dt = {}
  for i=0,255 do
   s = i/255
   dt[i+1] = gfx_dither_calc(grad,s)
  end
  add(dither_tables, dt)
 end
end

function gfx_dither( grad, s )
if grad > 3 then grad = 3 end
 return dither_tables[grad][1 + flr(s * 255)]
end

obj_cube = {}
obj_torus = {}
obj_fox = {}
sys_time = { t=0, dt=1/60, b=60 }

function _init()
 perf_counter( "bg" )
 perf_counter( "draw" )
 perf_counter( "update" )
 perf_counter( "vert" )
 perf_counter( "tri" )
 perf_counter( "shadow" )
 perf_counter( "drawlist" )
 
 perf_reset()

	obj_cube = obj_make_cube()
 obj_torus = obj_make_torus(0.75,0.5, 10, 6)
 obj_fox = obj_make_fox()
 init_dither()
end

function update()
 perf_begin("update")
 sys_time_tick()
 perf_end("update")
end


function _update60()
 update()
end

--function _update()
-- update()
--end


function obj_calc_normals()
	for t in all(obj.tri) do
		ova = obj.vtx[t[1]]
		ovb = obj.vtx[t[2]]
		ovc = obj.vtx[t[3]]

		dab = v3_sub( ovb, ova )
		dbc = v3_sub( ovc, ovb )

		t[5] = v3_normalize( v3_cross( dab, dbc ) )	
	end	
end

function obj_calc_bounds(obj)
	obj.bounds = {}

	fst = 1

	for v in all(obj.vtx) do
		if fst == 1 then
			vmin = v3(v[1], v[2], v[3])
			vmax = v3(v[1], v[2], v[3])
			fst = 0
		else
			vmin = v3_min( vmin, v )
			vmax = v3_max( vmax, v )
		end					
	end

	obj.bounds.c = v3_mul_s( v3_add( vmin, vmax ), .5 )
	obj.bounds.r = v3_length( v3_sub( vmax, vmin ) ) * .5
end

function obj_finalize(o)
	obj_calc_normals(o)
	obj_calc_bounds(o)
end

function obj_make_cube()
	obj = {}
	obj.vtx = {
  {-1, -1, -1},
  {1, -1, -1},
  {-1, 1, -1},
  {1, 1, -1},
  {1, -1, 1},
  {-1, -1, 1},
  {1, 1, 1},
  {-1, 1, 1},
 }

	obj.tri = { 
		{ 1, 2, 3, 1 },
		{ 2, 4, 3, 1 },
		{ 5, 6, 7, 1 },
		{ 6, 8, 7, 1 },
		{ 6, 1, 8, 2 },
		{ 8, 1, 3, 2 },
		{ 2, 5, 7, 2 },
		{ 2, 7, 4, 2 },
		{ 6, 5, 1, 3 },
		{ 5, 2, 1, 3 },
		{ 3, 7, 8, 4 },
		{ 3, 4, 7, 4 },
	}

 --obj.line = {
  --{ 1, 2, c = 7 },
  --{ 2, 4, c = 7 },
  --{ 4, 3, c = 7 },
  --{ 3, 1, c = 7 }
 --}
	
	obj_finalize(obj)

	return obj
end

function obj_make_fox()
 obj = {}
 obj.vtx = {
  {0.986111,-0.025855,0.300339},
  {-1.005903,-0.025855,0.300340},
  {-0.005903,0.000000,-3.763774},
  {-0.005903,1.000000,0.300339},
  {-0.005903,0.525322,1.004954},
  {-1.636132,-0.269847,-0.689093},
  {-1.874824,0.244348,0.182332},
  {-1.743159,1.274788,1.228072},
  {-1.264151,0.050001,0.029176},
  {-4.5,-1.035814,1.553706},
  {-1.189264,-0.440330,-1.404416},
  {-1.015540,0.175132,0.075046},
  {1.851038,0.244348,0.182331},
  {1.240366,0.050001,0.029176},
  {0.991755,0.175132,0.075046},
  {1.612347,-0.269847,-0.689093},
  {4.5,-1.035814,1.553705},
  {1.165478,-0.440330,-1.404416},
  {1.719373,1.274788,1.22807},
 }


 obj.tri = { 
{1,4,5,  5 }, --cowel left
{4,2,5,  6 }, --cowel right
{2,1,5,  9 }, --engine
{2,4,3,  7 }, --ship right
{1,2,3,  5 }, --ship bottom
{4,1,3,  6 }, --ship left
{6,7,2,  7 }, --right wing inside
{10,7,6, 6 }, --right wing front --hidden
{10,2,7, 6 }, --right wing top
{6,2,10, 5}, --right wing bottom
{11,9,8, 1},
{9,8,12, 1}, --right spoil top
{9,11,12, 1}, --right spoil side
{11,12,8, 1}, --right spoil front
{17,16,13, 6}, --right wing front
{18,19,14, 1 },
{14,19,15, 1 }, --left spoil top
{14,15,18, 1 }, --left spoil side
{18,15,19, 1 }, --left spoil bottom
{17,1,13, 6 }, --left wing back
{16,13,1, 6 }, 
{16,17,1, 5 } --left wing bottom
 }

 --obj.line = {
  --{ 1, 2, c = 7 },
  --{ 2, 4, c = 7 },
  --{ 4, 3, c = 7 },
  --{ 3, 1, c = 7 }
 --}
 
 obj_finalize(obj)

 return obj
end


function obj_make_torus(r0, r1, sweepsteps, steps)
 obj = {}
 obj.vtx = {}
 
 for step=0,steps-1 do
  stept = step / steps
  v = v3(0, sin(stept) * r1, cos(stept) * r1 + r0)

  for sweep=0,sweepsteps-1 do
   sweept = sweep / sweepsteps

   idx = step * sweepsteps + sweep + 1
   obj.vtx[idx] = v3_rot_y(v, sweept)
  end
 end

 obj.tri = {}
 for step=0,steps-1 do
  step1 = (step + 1) % steps
  for sweep=0,sweepsteps-1 do
   sweep1 = (sweep + 1) % sweepsteps

   i0 = 1+ step * sweepsteps + sweep
   i1 = 1+ step1 * sweepsteps + sweep
   i2 = 1+ step1 * sweepsteps + sweep1
   i3 = 1+ step * sweepsteps + sweep1
   add( obj.tri, {i0, i1, i2, 1 } ) 
   add( obj.tri, {i0, i2, i3, 1 } ) 
  end
 end  

 obj.lin = {}
 
 obj_finalize(obj)

 return obj
end

function tri_winding(ax,ay, bx,by, cx,cy)
 return (bx-ax)*(cy-by)-(by-ay)*(cx-bx)
end

function obj_draw( obj, obj_to_world, shadow )
 vs_set_obj_mat( obj_to_world )

	local scr_vtx = {}

 perf_begin("vert")
 local vtx = obj.vtx
 local vc = #vtx

 local sx,sy,ox,oy = vs_unpack_view_to_screen() 
 local r11,r12,r13, r21,r22,r23, r31,r32,r33, tx, ty, tz

 if not shadow then
  r11,r12,r13, r21,r22,r23, r31,r32,r33, tx, ty, tz = rt_unpack(vs.obj_to_cam)

  ldir = v3(0,1,0)
  obj_ldir = v3_mul_m3( ldir, vs.world_to_obj_rot )
 else
  r11,r12,r13, r21,r22,r23, r31,r32,r33, tx, ty, tz = rt_unpack(vs.obj_to_cam_shadow) 
 end

 for vi=1,vc do
  local ov = vtx[vi]
  local vx,vy,vz = rt_apply_unpacked( ov[1], ov[2], ov[3], r11,r12,r13, r21,r22,r23, r31,r32,r33, tx, ty, tz )
  scr_vtx[vi] = vs_view_to_screen_unpacked(vx,vy,vz,sx,sy,ox,oy)
 end

 perf_end("vert")


 local tc = #obj.tri

  local third = 1/3
 if not shadow then
  perf_begin("tri")
  for ti=1,tc do
   local t=obj.tri[ti]
   local a = scr_vtx[t[1]]
   local b = scr_vtx[t[2]]
   local c = scr_vtx[t[3]]

   local ax,ay,az = a[1],a[2],a[3]
   local bx,by,bz = b[1],b[2],b[3]
   local cx,cy,cz = c[1],c[2],c[3]

   local nr = vs.near
   -- backface cull
   if ((bx-ax)*(cy-by)-(by-ay)*(cx-bx)) < 0.0 then
    if az > nr and bz > nr and cz > nr then

     local s = mid( v3_dot(obj_ldir, t[5]) * -0.5 + 0.5, 0, 1)
     local d = gfx_dither( t[4], s )  

     add( drawlist, { key=(az + bz + cz) * third, fn = dl_tri, value = {a=a, b=b, c=c, col=d.c, fp=d.f } } )
     --fillp(d.f)
     --trifill( a, b, c, d.c )
   end
   end
  end
  perf_end("tri")

 else
  perf_begin("shadow")
  
  fillp(0b0101101001011010.1)
  for ti=1,tc do
   local t=obj.tri[ti]
   local a=scr_vtx[t[1]]
   local b=scr_vtx[t[2]]
   local c=scr_vtx[t[3]]
   local ax,ay,az = a[1],a[2],a[3]
   local bx,by,bz = b[1],b[2],b[3]
   local cx,cy,cz = c[1],c[2],c[3]

   local nr = vs.near
   -- backface cull
   if ((bx-ax)*(cy-by)-(by-ay)*(cx-bx)) < 0.0 then
    if az > nr and bz > nr and cz > nr then
     trifill( ax,ay,bx,by,cx,cy, 0 )
    end
   end
  end
  perf_end("shadow")

 end

 if obj.lin then
  local lc = #obj.lin
  for li=1,lc do
   local l=obj.lin[li]
   local a = scr_vtx[l[1]]
   local b = scr_vtx[l[2]]
   if a[3] > vs.near and b[3] > vs.near then
    if shadow then
     fillp(0b0101101001011010.1)
     line(a[1], a[2], b[1], b[2], 0)
    else
     local key = (a[3] + b[3]) / 2
     add( drawlist, { key=key, fn = dl_line, value = {a=a, b=b, col=l.c } } )
    end
   end
  end
 end

 if not shadoe then
  perf_begin("drawlist")
  dl_draw()
  dl_reset()
  perf_end("drawlist")
 end
end


-- scene

scene = {}

function scene_reset()
 scene = {}
end


function scene_draw( bg )
 if bg then
  for item in all(scene) do
   if bg then
    if item.value.bg then item.value.draw(item.value, bg) end
   end
  end
 else
  local sortlist = {}

  for k,v in pairs(scene) do
   add(sortlist, {key=-v.key,sc_key=k})
  end
  
  ce_heap_sort(sortlist)

  for p in all(sortlist) do
   local item = scene[p.sc_key]
   if item.value.fg then 
    item.value.draw(item.value, bg) 
   end
  end

  if not (band(dbg,1) == 0) then
   local ic = #scene
   for ii=1,ic do
    local item = scene[ii]
    local str = "key:" .. item.key
    gfx_3d_print(item.value.wp, str, 7)
   end
  end
 end 
end

function scene_key(wp)
  local vp = v3_sub(wp, vs.cam_to_world.t)  
  return v3_dot( vp, vs.cam_z )
end  


function scene_add_obj( obj, obj_to_world )
 local t = obj_to_world.t
 local rt = {}
 rt.t = v3(t[1], t[2], t[3])
 local r = obj_to_world.r
 rt.r = { {r[1][1], r[1][2], r[1][3]}, {r[2][1], r[2][2], r[2][3]}, {r[3][1], r[3][2], r[3][3]} }

 local bwc = rt_apply( obj.bounds.c, obj_to_world ) 
 if vs_cull_sphere( bwc, obj.bounds.r ) then
  local key = scene_key(bwc)
  add( scene,
  { 
   key = key, 
   value = {
    draw = scene_draw_obj,
    bg = true,
    fg = true,   
    wp = v3(bwc[1], bwc[2], bwc[3]),
    obj = obj,
    rt = rt
   } 
  } )
 end
end

function scene_add_sprite( p, spr_def )
 local max_r = max( spr_def.w, spr_def.h )
 if vs_cull_sphere( p, max_r ) then
  local key = scene_key(p)
  
  add( scene,
  { 
   key = key, 
   value = {
    draw = scene_draw_sprite,
    bg = false,
    fg = true,   
    wp = v3(p[1], p[2], p[3]),
    s = spr_def
   } 
  } )
 end
end

spr_def = 
{
 w = 0.5,
 h = 0.5,
 sx = 40,
 sy = 0, 
 sw = 16,
 sh = 16
}

function scene_build()

 scene_reset()

 add( scene,
 { 
  key = -32767, 
  value = {
   draw = scene_draw_background,
   bg = true,
   fg = false,
   wp = v3(0,0,0),   
  } 
 } )

 local y_rot = sys_time.t * 1
 local x_rot = sys_time.t * 0.234

 local obj_r1 = m3_rot_y(y_rot)
 local obj_r2 = m3_rot_x(x_rot)

 local obj_to_world =
  { r=m3_mul( obj_r2, obj_r1 ), 
    t=v3( 0, 5, 0 ) }

   --gfx_3d_sphere_outline( rt_apply( cube.bounds.c, obj_to_world ), cube.bounds.r )
   -- gfx_3d_sprite( rt_apply( cube.bounds.c, obj_to_world ), cube.bounds.r, cube.bounds.r * 0.75, 8, 0, 16, 16 )
   
   for z=3,-3,-1 do
    scene_add_sprite( v3(4,0.5,z * 4), spr_def )
    --scene_add_sprite( v3(-4,0.5,z * 4), spr_def )
   end

   --scene_add_obj( obj_cube, obj_to_world )
   --scene_add_obj( obj_fox, obj_to_world )
   scene_add_obj( obj_torus, obj_to_world )
 
   for x=2,5 do
     obj_to_world.t[1] = x * 4
     scene_add_obj( obj_cube, obj_to_world )
   end    


end

function vgrad(y0, y1, i0, i1, g)
 local s = i0
 local ds_dy = (i1 - i0) / (y1 - y0)
 for y=y0,y1 do
  d = gfx_dither( g, s )
  fillp(d.f)
  rect(0,y, 127,y, d.c)

  s += ds_dy
 end
end

function scene_draw_background()
  perf_begin("bg")
  draw_floor()
  fillp()

  gfx_3d_grid(6)
  perf_end("bg")
end

function scene_draw_obj( scene_obj, bg )
 obj_draw( scene_obj.obj, scene_obj.rt, bg )
end

function scene_draw_sprite( val, bg ) 
 gfx_3d_sprite( val.wp, val.s.w, val.s.h, val.s.sx, val.s.sy, val.s.sw, val.s.sh )
end

function draw_floor()
 local d = 1000
 local za = vs.cam_to_world.r[3]
 local z = v3(za[1],za[2],za[3])
 z[2] = 0
 z = v3_normalize(z)
 pw = v3_add(v3_mul_s(z,d), vs.cam_to_world.t)
 pw[2] = 0
 
 local vv = rt_apply( pw, vs.world_to_cam )
 vv = vs_view_to_screen( vv )

 y = vv[2]
 --if ( y <= 127 ) rectfill(0,0,127,y,1)
 --if ( y > 0 ) rectfill(0,y,127,127,3)
 --light = 0
 --vgrad(0,y, 0.6 + light * 0.3, 0.1 + light * 0.05 , 1 )
 --vgrad(y+1, 127, 0.1 + light * 0.05, 0.6+ light * 0.2, 2 )

 local vcam = v3(0,0,0)
 local vtop = v3(0,vs.vp.tl[2], vs.pdist )
 local vbot = v3(0,vs.vp.br[2], vs.pdist )

 vcam = rt_apply( vcam, vs.cam_to_world )
 vtop = rt_apply( vtop, vs.cam_to_world )
 vbot = rt_apply( vbot, vs.cam_to_world )

 local fl_h = 0
 local cam_h = vcam[2] - fl_h

 local vtopd = v3_sub( vtop, vcam )
 local vbotd = v3_sub( vbot, vcam )
 local vtopd_y = vtopd[2]
 local vbotd_y = vbotd[2]

 vtopd_xz = sqrt( vtopd[1] * vtopd[1] + vtopd[3] * vtopd[3] )
 vbotd_xz = sqrt( vbotd[1] * vbotd[1] + vbotd[3] * vbotd[3] )

 -- y = ground y
 -- z = ground intersect z
 -- oy, oz = ray o
 -- dy, dz = ray d

 --(1) y = oy + dy * t
 --(2) z = oz + dz * t

 --(3) t = (y - oy) / dy 
 --z = oz + dz * (y - oy) / dy

 local d_y = vtopd_y
 local d_xz = vtopd_xz
 local d_y_dy = (vbotd_y - vtopd_y) / 128
 local d_xz_dy = (vbotd_xz - vtopd_xz) / 128

 for y=0,127 do
  t = cam_h / -d_y
  local c = 2
  local s = 1
  if t > 0 then  
   local xz = d_xz * t * s
   c = 2
   s = (1 - mid( 1 / xz, 0, 1)) * .2
  else
   c = 1
   s = (1 - mid( d_y, 0, 1 )) * .8
  end
  di = gfx_dither( c, s )
  fillp(di.f)

  rectfill(0,y,127,y,di.c)
  
  d_y += d_y_dy
  d_xz += d_xz_dy
 end
end

cam_pos = v3(0,1,-10)
cam_angles = v3(0,0,0)

function _draw()
  perf_reset()
  perf_begin("draw")

	--cls()

	--map(0,0, 0,0, 16,16)

	-- trifill( v2(10, 30), v2(50, 40), v2(25, 100), 0x1111 )

-- drawlist

dl_reset()


   cam_move = v3(0,0,0)
      if (btn(4)) then
       if ( btn(0) )cam_move[1]-=.1
       if ( btn(1) )cam_move[1]+=.1

       if ( btn(2) )cam_move[3]+=.1
       if ( btn(3) )cam_move[3]-=.1
      else
       if ( btn(0) )cam_angles[2]-=.01
       if ( btn(1) )cam_angles[2]+=.01

       if ( btn(2) )cam_angles[1]+=.01
       if ( btn(3) )cam_angles[1]-=.01
      end

   cam_m = m3_mul( m3_rot_y(cam_angles[2]), m3_rot_x(cam_angles[1]) )

      cam_pos = v3_add( cam_pos, v3_mul_m3(cam_move, cam_m) )

   cam_to_world =
    { r=cam_m,
      t= cam_pos }
      --v3(0,1,-10 - 8. * sin(fr * 0.001)) }

   pdist = 1.0
   vp = { tl = v2(-1,1), br = v2(1,-1) }
   win = { tl = v2(0,0), br = v2(128,128) }
   vs_view_setup( cam_to_world, vp, pdist, win )

if 0 == 0 then 
	
	color(15)
 fillp()



 --rectfill(0,0,127,64, 1)
 --rectfill(0,128,127,64, 3)

 scene_build()

 scene_draw( true )
 scene_draw( false )


 --gfx_3d_print(v3(0, 4, 0), "hello", 7)

   --gfx_3d_line( v3(0,0,0), v3(3, 0, 0), 4)
end

 perf_end("draw")

 perf_hud()
end

__gfx__
0000000099aaaaaaaaa777aab3b33333444444440000122a4aa00000000000000000000000000000000000000000000000000000000000000000000000000000
00000000999999999999999ab3b333b349442494000129984444a200000000000000000000000000000000000000000000000000000000000000000000000000
0070070099a444222220209733333333444444440022989989899421000000000000000000000000000000000000000000000000000000000000000000000000
0007700099a99a499a4992973b333333424444440129999aa9a988a2000000000000000000000000000000000000000000000000000000000000000000000000
0007700049a99a499a4990973b33b3b344494244124899aaaaa99892000000000000000000000000000000000000000000000000000000000000000000000000
0070070049a99a499a49909a333333334444444424989a777aaaa998000000000000000000000000000000000000000000000000000000000000000000000000
0000000029799a499a49909a33b33b3394244424a4899a77777aa998000000000000000000000000000000000000000000000000000000000000000000000000
0000000029799a499a49929a33333b3344444444a4899a777777a999000000000000000000000000000000000000000000000000000000000000000000000000
0000000049a99a499a49929a0000000000000000a4899a777aaaa998000000000000000000000000000000000000000000000000000000000000000000000000
0000000049a99a499a49929a000000000000000024899a77aaaa9984000000000000000000000000000000000000000000000000000000000000000000000000
0000000049a99a499a49949a000000000000000024889aaaaaaa9482000000000000000000000000000000000000000000000000000000000000000000000000
0000000029799a499a49949a00000000000000001249999999944881000000000000000000000000000000000000000000000000000000000000000000000000
0000000029a99a499a49949900000000000000001224888899998921000000000000000000000000000000000000000000000000000000000000000000000000
0000000029aa7aaaa7aa7a9900000000000000000122444488882900000000000000000000000000000000000000000000000000000000000000000000000000
00000000499999999999999900000000000000000011222222221100000000000000000000000000000000000000000000000000000000000000000000000000
00000000444222244449999900000000000000000000111221110000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

__gff__
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
__map__
0303030303030303030303030303030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0404030303030403030303030303030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0404030303030404040303030102040400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0403040404030303030403041112030400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0304040403030303030304030304030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0304030404040304040403030404030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0303030303030303040303030303030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0303030303030404040304040303030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0303030303030304040403040303030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0403030303030304030403030303030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0404030304030303030303030303030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0404030404040303030303030403030400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0403040404030303040304030303030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0303030303040304030303040303030400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0303030303030303030303030303030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0303030403030303030404040404030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0303040404040403030304040404040300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0303040403030303030303040403030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0303030303030303030303030303030300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000040300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
__sfx__
000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
__music__
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344
00 41424344

