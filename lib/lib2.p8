pico-8 cartridge // http://www.pico-8.com
version 14
__lua__

dbg = 0
perf = 0

----------------------
-- system
----------------------

dbg = 0
perf = true

perf_counters = {}

-- misc

function ceil(num)
 return flr(num+0x0.ffff)
end

function sat(num)
 return mid(num, 0, 1)
end

function min3(a,b,c)
 return min(a, min(b,c))
end

function max3(a,b,c)
 return max(a, max(b,c))
end

sys_time = { t=0, dt=1/60, b=60 }

function sys_time_tick(framerate)
 sys_time.b=framerate
 sys_time.dt=1/framerate
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
 local fps=sys_time.b/ceil(stat(1))
 local mem=flr(stat(0))
 local perf=
  cpu .. "% cpu @ " ..
  fps ..  " fps " ..
  mem .. " mb"
 print(perf,0,122,0)
 print(perf,0,121,fps==sys_time.b and 7 or 8)

 perf_draw_timers()
end

-- vector2

function v2_unpack(v)
 return v[1],v[2]
end

function v2_tostring(x,y)
 return "(" .. x .. "," .. y .. ")"
end

function v2_neg(x,y)
 return -x,-y
end

function v2_rcp(x,y)
 return 1/x,1/y
end

function v2_add(ax,ay,bx,by)
 return ax+bx,ay+by
end

function v2_add_s(ax,ay,b)
 return ax+b,ay+b
end

function v2_sub(ax,ay,bx,by)
 return ax-bx,ay-by
end

function v2_sub_s(ax,ay,b)
 return ax-b,ay-b
end

function v2_mul(ax,ay,bx,by)
 return ax * bx, ay * by
end

function v2_mul_s(ax,ay,b)
 return ax * b, ay * b
end

function v2_cross(ax,ay,bx,by)
 return ax*by-ay*bx
end

-- vector3 

function v3_unpack(v)
 return v[1],v[2],v[3]
end

function v3_tostring(x,y,z)
 return "(" .. x .. "," .. y .. "," .. z .. ")"
end

function v3_neg(x,y,z)
 return -x,-y,-z
end

function v3_rcp(x,y,z)
 return 1/x,1/y,1/z
end

function v3_add(ax,ay,az,bx,by,bz)
 return ax+bx,ay+by,az+bz
end

function v3_add_s(ax,ay,az,b)
 return ax+b,ay+b,az+b
end

function v3_sub(ax,ay,az,bx,by,bz)
 return ax-bx,ay-by,az-bz
end

function v3_sub_s(ax,ay,az,b)
 return ax-b,ay-b,az-b
end

function v3_mul(ax,ay,az,bx,by,bz)
 return ax*bx,ay*by,az*bz
end

function v3_mul_s(ax,ay,az,b)
 return ax*b,ay*b,az*b
end

function v3_dot(ax,ay,az,bx,by,bz)
 return ax*bx + ay*by + az*bz
end

function v3_cross(ax,ay,az,bx,by,bz)
 return ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx
end

function v3_min(ax,ay,az,bx,by,bz)
 return min(ax,bx), min(ay,by), min(az,bz)
end

function v3_max(ax,ay,az,bx,by,bz)
 return max(ax,bx), max(ay,by), max(az,bz)
end

function v3_length(x,y,z)
 return sqrt( v3_dot(x,y,z,x,y,z) )
end 

function v3_normalize(x,y,z)
 local rl = 1 / v3_length(x,y,z)
 return x*rl, y*rl, z*rl
end

-- plane

function pl_abc(ax,ay,az,bx,by,bz,cx,cy,cz)
 local nx,ny,nz = v3_normalize( v3_cross(v3_sub(ax,ay,az, bx,by,bz), v3_sub(cx,cy,cz, bx,by,bz)) )
 return nx,ny,nz,-v3_dot(nx,ny,nz, ax,ay,az)
end

function pl_dist(px,py,pz,pd, vx,vy,vz)
 return px*vx + py*vy + pz*vz + pd
end

-- matrix

function m3_unpack(m)
 return m[1],m[2],m[3], m[4],m[5],m[6], m[7],m[8],m[9]
end

function m3_id()
 return 1,0,0, 0,1,0, 0,0,1
end 

function m3_get_ax(m)
 return m[1], m[4], m[7]
end
function m3_get_ay(m)
 return m[2], m[5], m[8]
end

function m3_get_az(m)
 return m[3], m[6], m[9]
end

function m3_rot_x(t)
 local s = sin(t)
 local c = cos(t)
 return 1,0,0, 0,c,s, 0,-s,c
end

function m3_rot_y(t)
 local s = sin(t)
 local c = cos(t)
 return c,0,-s, 0,1,0, s,0,c
end

function m3_rot_z(t)
 local s = sin(t)
 local c = cos(t)
 return c,-s,0, s,c,0, 0,0,1
end

function m3_mul(a11,a12,a13, a21,a22,a23, a31,a32,a33, b11,b12,b13, b21,b22,b23, b31,b32,b33)
 -- aik * bkj + aik * bkj + aik * bkj
 return 
  a11 * b11 + ai2 * b21 + a13 * b31,
  a11 * b12 + ai2 * b22 + a13 * b32,
  a11 * b13 + ai2 * b23 + a13 * b33,
  a21 * b11 + ai2 * b21 + a23 * b31,
  a21 * b12 + ai2 * b22 + a23 * b32,
  a21 * b13 + ai2 * b23 + a23 * b33,
  a31 * b11 + ai2 * b21 + a33 * b31,
  a31 * b12 + ai2 * b22 + a33 * b32,
  a31 * b13 + ai2 * b23 + a33 * b33
end 

function m3_trans(m11,m12,m13, m21,m22,m23, m31,m32,m33)
 return m11,m21,m31, m12,m22,m32, m13,m23,m33
end


function v3_mul_m3(vx,vy,vz, m11,m12,m13, m21,m22,m23, m31,m32,m33)
 return
  vx * m11 + vy * m12 + vz * m13,
  vx * m21 + vy * m22 + vz * m23,
  vx * m31 + vy * m32 + vz * m33
end

-- rot-trans

function rt_unpack(rt)
 return m3_unpack(rt.r), v3_unpack(rt.t)
end

function rt_apply(vx,vy,vz, r11,r12,r13, r21,r22,r23, r31,r32,r33, tx, ty, tz )
 return 
  tx + vx * r11 + vy * r12 + vz * r13,
  ty + vx * r21 + vy * r22 + vz * r23,
  tz + vx * r31 + vy * r32 + vz * r33
end

function rt_mul(rta, rtb)
 return {
  r = { m3_mul(m3_unpack(rta.r), m3_unpack(rtb.r)) },
  t = { v3_add(v3_unpack(rtb.t), v3_mul_m3(v3_unpack(rta.t), m3_unpack(rtb.r))) }
 }
end

function rt_inv(rt)
 local r = { m3_trans( m3_unpack(rt.r) ) }
 return { r=r, t= { v3_mul_m3(v3_neg( v3_unpack(rt.t) ), r ) } }
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
   vs.world_to_obj_rot = m3_trans( obj_to_world.r )
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

 slc = #sortlist
 for i=1,slc do
  p = sortlist[i]
  item = drawlist[p.dl_key]
  item.fn( item.value )
 end
end


function dl_tri(v)
   fillp(v.fp)
   gfx_tri_fill( v.a, v.b, v.c, v.col )
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


function gfx_tri_wire( a, b, c )
	line(a[1], a[2], b[1], b[2])
	line(b[1], b[2], c[1], c[2])
	line(c[1], c[2], a[1], a[2])
end

function hline( xl, xr, y, col )
	--if y < vs.sc.tl.y or y >= vs.sc.br.y then return end

	if xr < xl then xl,xr = xr,xl end
	xl = flr(xl)
	xr = flr(xr)
	xl = mid( xl, vs.scl, vs.scr )
	xr = mid( xr, vs.scl, vs.scr )

	x = xl
	w = xr - xl
	
	--hspan( xl, y, xr-xl, cc )

	if (w <= 0) return

	scan = 0x6000 + y * 64
	addr = scan + flr( x / 2 )

	if x % 2 == 1 then
		val = peek( addr )
		poke( addr, band(val, 0x0f) + band(cc, 0xf0))
		w-=1
		addr += 1
	end

	b = flr(w / 2)
	memset( addr, cc, b )

	w -= b * 2
	addr += b

	if w > 0 then
		val = peek( addr )
		poke( addr, band(val, 0xf0) + band(cc, 0x0f))
	end	
end

function sort_y_3( a, b, c )
 if a[2] > b[2] then
  a,b = b,a
 end

 if b[2] > c[2] then
  b,c = c,b
  if a[2] > b[2] then
   a,b = b,a
  end  
 end
 return a,b,c
end

function gfx_tri_fill( a, b, c, col )

	-- sort vertices
 a,b,c = sort_y_3( a, b, c )

 -- ceil and clip y
 ayc = mid( ceil(a[2]), vs.sct, vs.scb )
 byc = mid( ceil(b[2]), vs.sct, vs.scb )
 cyc = mid( ceil(c[2]), vs.sct, vs.scb )

 -- init edges
	dabx_dy = (b[1] - a[1]) / (b[2] - a[2])
	dacx_dy = (c[1] - a[1]) / (c[2] - a[2])
 dbcx_dy = (c[1] - b[1]) / (c[2] - b[2])

	ab_x = a[1] + (ayc - a[2]) * dabx_dy
	ac_x = a[1] + (ayc - a[2]) * dacx_dy
 bc_x = b[1] + (byc - b[2]) * dbcx_dy

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
 obj_torus = obj_make_torus(0.75,0.5, 6, 6)
 obj_fox = obj_make_fox()
 init_dither()
end

function update(dt)
 perf_start("update")
 sys_time.dt = dt
 sys_time.t += dt
 perf_end("update")
end


function _update60()
 sys_time.b=60
 update(1/sys_time.b)
end

--function _update()
-- sys_time.b=30
-- update(1/sys_time.b)
--end


function obj_calc_normals()
	for t in all(obj.tri) do
		local ovax, ovay, ovaz = v3_unpack( obj.vtx[t[1]] )
		local ovbx, ovby, ovbz = v3_unpack( obj.vtx[t[2]] )
		local ovcx, ovcy, ovcz = v3_unpack( obj.vtx[t[3]] )

		local dabx,daby,dabz = v3_sub( ovbx,ovby,ovbz, ovax,ovay,ovaz )
		local dbcx,dbcy,dbcz = v3_sub( ovcx,ovcy,ovcz, ovbx,ovby,ovbz )

  local cpx,cpy,cpz = v3_cross( dabx,daby,dabz, dbcx,dbcy,dbcz )
		t.n = { v3_normalize( cpx,cpy,cpz )	}
	end	
end

function obj_calc_bounds(obj)
	obj.bounds = {}

	fst = true

 local vminx, vminy, vminz
 local vmaxx, vmaxy, vmaxz

	for v in all(obj.vtx) do
  local vx,vy,vz = v3_unpack(v)
		if fst then
			vminx,vminy,vminz = vx,vy,vz
   vmaxx,vmaxy,vmaxz = vx,vy,vz
			fst = false
		else
			vminx,vminy,vminz = v3_min( vminx,vminy,vminz, vx,vy,vz )
   vmaxx,vmaxy,vmaxz = v3_max( vmaxx,vmaxy,vmaxz, vx,vy,vz )
		end					
	end

	obj.bounds.c = { v3_mul_s( v3_add( vminx,vminy,vminz, vmaxx,vmaxy,vmaxz ), .5 ) }
	obj.bounds.r = { v3_length( v3_sub( vmaxx,vmaxy,vmaxz, vminx,vminy,vminz ) ) * .5 }
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
		{ 1, 2, 3, c = 1 },
		{ 2, 4, 3, c = 1 },
		{ 5, 6, 7, c = 1 },
		{ 6, 8, 7, c = 1 },
		{ 6, 1, 8, c = 2 },
		{ 8, 1, 3, c = 2 },
		{ 2, 5, 7, c = 2 },
		{ 2, 7, 4, c = 2 },
		{ 6, 5, 1, c = 3 },
		{ 5, 2, 1, c = 3 },
		{ 3, 7, 8, c = 4 },
		{ 3, 4, 7, c = 4 },
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
{ 0.986111,-0.025855,0.300339 },
{ -1.005903,-0.025855,0.300340 },
{ -0.005903,0.000000,-3.763774 },
{ -0.005903,1.000000,0.300339 },
{ -0.005903,0.525322,1.004954 },
{ -1.636132,-0.269847,-0.689093 },
{ -1.874824,0.244348,0.182332 },
{ -1.743159,1.274788,1.228072 },
{ -1.264151,0.050001,0.029176 },
{ -4.5,-1.035814,1.553706 },
{ -1.189264,-0.440330,-1.404416 },
{ -1.015540,0.175132,0.075046 },
{ 1.851038,0.244348,0.182331 },
{ 1.240366,0.050001,0.029176 },
{ 0.991755,0.175132,0.075046 },
{ 1.612347,-0.269847,-0.689093 },
{ 4.5,-1.035814,1.553705 },
{ 1.165478,-0.440330,-1.404416 },
{ 1.719373,1.274788,1.22807 }
 }


 obj.tri = { 
{1,4,5,  c=5 }, --cowel left
{4,2,5,  c=6 }, --cowel right
{2,1,5,  c=9 }, --engine
{2,4,3,  c=7 }, --ship right
{1,2,3,  c=5 }, --ship bottom
{4,1,3,  c=6 }, --ship left
{6,7,2,  c=7 }, --right wing inside
{10,7,6, c=6 }, --right wing front --hidden
{10,2,7, c=6 }, --right wing top
{6,2,10, c=5}, --right wing bottom
{11,9,8, c=1},
{9,8,12, c=1}, --right spoil top
{9,11,12, c=1}, --right spoil side
{11,12,8, c=1}, --right spoil front
{17,16,13, c=6}, --right wing front
{18,19,14, c=1 },
{14,19,15, c=1 }, --left spoil top
{14,15,18, c=1 }, --left spoil side
{18,15,19, c=1 }, --left spoil bottom
{17,1,13, c=6 }, --left wing back
{16,13,1, c=6 }, 
{16,17,1, c=5 } --left wing bottom
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
   add( obj.tri, {i0, i1, i2, c=1 } ) 
   add( obj.tri, {i0, i2, i3, c=1 } ) 
  end
 end  

 obj.lin = {}
 
 obj_finalize(obj)

 return obj
end

function transform_vert_shadow( ov )
   local vw = rt_apply( ov, vs.obj_to_world )   
   vw[1] += vw[2] * 0.2
   vw[2] = 0
   return vs_view_to_screen( rt_apply( vw, vs.world_to_cam ) )
end

function obj_draw( obj, obj_to_world, shadow )
 vs_set_obj_mat( obj_to_world )

	local scr_vtx = {}

 perf_start("vert")
 if not shadow then
  local vc = #obj.vtx
 	for vi=1,vc do
 	 scr_vtx[vi] = vs_view_to_screen( rt_apply( obj.vtx[vi], vs.obj_to_cam ) )
 	end

  ldir = v3(0,1,0)
  obj_ldir = v3_mul_m3( ldir, vs.world_to_obj_rot )
 else
  local vc = #obj.vtx
 	for vi=1,vc do
   scr_vtx[vi] = transform_vert_shadow(obj.vtx[vi])
  end
 end

 perf_end("vert")


 local tc = #obj.tri

 if not shadow then
  perf_start("tri")
  for ti=1,tc do
   local t=obj.tri[ti]
   local a = scr_vtx[t[1]]
   local b = scr_vtx[t[2]]
   local c = scr_vtx[t[3]]

   -- backface cull
   if v2_cross( v2_sub( b, a ), v2_sub( c, b ) ) < 0.0 then
    if a[3] > vs.near and b[3] > vs.near and c[3] > vs.near then

     local col, fp

     local ldotn = v3_dot(obj_ldir, t.n)
     local s = sat( ldotn * -0.5 + 0.5)
     local d = gfx_dither( t.c, s )  

     local key = (a[3] + b[3] + c[3]) / 3
     add( drawlist, { key=key, fn = dl_tri, value = {a=a, b=b, c=c, col=d.c, fp=d.f } } )
     --fillp(d.f)
     --gfx_tri_fill( a, b, c, d.c )
   end
   end
  end
  perf_end("tri")

 else
  perf_start("shadow")
  
  fillp(0b0101101001011010.1)
  for ti=1,tc do
   local t=obj.tri[ti]
   local a = scr_vtx[t[1]]
   local b = scr_vtx[t[2]]
   local c = scr_vtx[t[3]]

   -- near cull
   if a[3] > vs.near and b[3] > vs.near and c[3] > vs.near then
    -- backface cull
    if v2_cross( v2_sub( b, a ), v2_sub( c, b ) ) < 0.0 then
     gfx_tri_fill( a, b, c, 0 )
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

 perf_start("drawlist")

 dl_draw()
 dl_reset()
 perf_end("drawlist")

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
    --scene_add_sprite( v3(4,0.5,z * 4), spr_def )
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
  perf_start("bg")
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
 z[1],z[2],z[3] = v3_normalize(z[1], z[2], z[3])
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

cam_pos = {0,1,-10}
cam_angles = {0,0,0}

function _draw()
  perf_reset()
  perf_start("draw")

	--cls()

	--map(0,0, 0,0, 16,16)

	-- gfx_tri_fill( v2(10, 30), v2(50, 40), v2(25, 100), 0x1111 )

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

	perf_draw()
 perf_draw_timers()
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

