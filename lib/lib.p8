pico-8 cartridge // http://www.pico-8.com
version 12
__lua__

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

-- vector2

function v2(x,y)
 return {x=x,y=y}
end

function v2_tostring(a)
 return "(" .. a.x .. "," .. a.y .. ")"
end

function v2_neg(a)
 return v2(-a.x, -a.y)
end

function v2_rcp(a)
 return v2(1/a.x, 1/a.y)
end

function v2_add(a,b)
 return v2(a.x+b.x, a.y+b.y)
end

function v2_add_s(a,b)
 return v2(a.x+b, a.y+b)
end

function v2_sub(a,b)
 return v2(a.x-b.x, a.y-b.y)
end

function v2_mul(a, b)
	return v2( a.x * b.x, a.y * b.y )
end

function v2_mul_s(a, b)
    return v2( a.x * b, a.y * b )
end

function v2_cross(a,b)
	return a.x*b.y-a.y*b.x
end

-- vector3 

function v3(x,y,z)
	return {x=x,y=y,z=z}
end

function v3_tostring(a)
	return "(" .. a.x .. "," .. a.y .. "," .. a.z .. ")"
end

function v3_neg(a)
	return v3(-a.x, -a.y, -a.z)
end

function v3_rcp(a)
 return v3(1/a.x, 1/a.y, 1/a.z)
end

function v3_add(a,b)
	return v3(a.x+b.x, a.y+b.y, a.z+b.z)
end

function v3_add_s(a,b)
	return v3(a.x+b, a.y+b, a.z+b)
end

function v3_sub(a,b)
	return v3(a.x-b.x, a.y-b.y, a.z-b.z)
end

function v3_sub_s(a,b)
	return v3(a.x-b, a.y-b, a.z-b)
end

function v3_mul(a,b)
	return v3(a.x*b.x,a.y*b.y,a.z*b.z)
end

function v3_mul_s(a, b)
	return v3( a.x*b, a.y*b, a.z*b )
end

function v3_dot(a, b)
	return a.x * b.x + a.y * b.y + a.z * b.z
end

function v3_cross(a, b)
	return v3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x )
end

function v3_min(a,b)
	return v3( min(a.x,b.x), min(a.y,b.y), min(a.z,b.z) )
end

function v3_max(a,b)
	return v3( max(a.x,b.x), max(a.y,b.y), max(a.z,b.z) )
end

function v3_length(a)
	return sqrt( v3_dot(a,a) )
end	

function v3_normalize(a)
	return v3_mul_s( a, 1 / v3_length(a) )
end

function v3_rot_x(a, t)
	s = sin(t)
	c = cos(t)
	return v3( a.x, a.y * c + a.z * s, a.y * -s + a.z * c )	
end

function v3_rot_y(a, t)
	s = sin(t)
	c = cos(t)
	return v3( a.x * c + a.z * s, a.y, a.x * -s + a.z * c )	
end

-- plane
function pl(n,d)
 return { x=n.x, y=n.y, z=n.z, d=d }
end

function pl_abc(a,b,c)
 local ba = v3_sub(a,b)
 local bc = v3_sub(c,b)
 local n = v3_normalize( v3_cross(ba, bc) )
 return pl( n, -v3_dot(n, a) )
end

function pl_dist(p, v)
 return v3_dot(p,v) + p.d
end

-- matrix

function m3(x,y,z)
	return {{x.x,x.y,x.z},{y.x,y.y,y.z},{z.x,z.y,z.z}}
end	

function m3_id()
	return {{1,0,0},{0,1,0},{0,0,1}}
end	

function m3_get_ax(m)
 return v3(m[1][1], m[1][2], m[1][3])
end
function m3_get_ay(m)
 return v3(m[2][1], m[2][2], m[2][3])
end

function m3_get_az(m)
 return v3(m[3][1], m[3][2], m[3][3])
end

function m3_rot_x(t)
	s = sin(t)
	c = cos(t)
	return m3(v3(1, 0, 0), v3(0, c,  s), v3(0,  -s,  c))
end

function m3_rot_y(t)
	s = sin(t)
	c = cos(t)
	return m3(v3(c, 0, -s), v3(0,  1,  0), v3(s,  0,  c))
end

function m3_rot_z(t)
	s = sin(t)
	c = cos(t)
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
	return v3( v.x * m[1][1] + v.y * m[1][2] + v.z * m[1][3],
	 v.x * m[2][1] + v.y * m[2][2] + v.z * m[2][3],
	 v.x * m[3][1] + v.y * m[3][2] + v.z * m[3][3] )
end

-- rot-trans

function rt_apply(v, rt)
	return v3_add( v3_mul_m3(v,rt.r), rt.t)
end

function rt_mul(rta, rtb)
 local rt = {}
 rt.r = m3_mul(rta.r, rtb.r)
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
 for p in all(vs.frustum.planes) do
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
 vptl = v3(vs.vp.tl.x, vs.vp.tl.y, vs.pdist)
 vptr = v3(vs.vp.br.x, vs.vp.tl.y, vs.pdist)
 vpbr = v3(vs.vp.br.x, vs.vp.br.y, vs.pdist)
 vpbl = v3(vs.vp.tl.x, vs.vp.br.y, vs.pdist)

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
 vs.sc = sc
 if (sc == nil) vs.sc = win

 vs.sct = vs.sc.tl.y
 vs.scb = vs.sc.br.y - 1
 vs.scl = vs.sc.tl.x
 vs.scr = vs.sc.br.x - 1

 vs.vp.size = v2_sub( vs.vp.br, vs.vp.tl )  
 vs.vp.rcp_size = v2_rcp( vs.vp.size ) 

 vs.win.size = v2_sub( vs.win.br, vs.win.tl )

 -- viewport to screen

 -- transform to viewport
 --pv = v2_sub( pp, vs.vp.tl ) -- a
 --pv = v2_mul( pv, vs.vp.rcp_size ) -- b

-- transform to screen
 --ps = v2_mul( pv, vs.win.size ) -- c
 --ps = v2_add( ps, vs.win.tl ) -- d

-- r = x * bc + d - a * bc
vs.st_scale = v2_mul( vs.vp.rcp_size, vs.win.size )
vs.st_offset = v2_mul( v2_sub( vs.win.tl, vs.vp.tl ), vs.st_scale )


 vs.cam_to_world = cam_to_world
 vs.world_to_cam = rt_inv( cam_to_world )

 vs_frustum_setup()
end

function vs_set_obj_mat( obj_to_world )
   vs.obj_to_world = obj_to_world
   vs.obj_to_cam = rt_mul( obj_to_world, vs.world_to_cam )
   vs.world_to_obj_rot = m3_trans( obj_to_world.r )
end


function vs_view_to_screen( v )
 -- project onto screen at distance vs.pdist in view space
 local w = 1. / v.z
 local ps = v2_add(v2_mul(v2_mul_s( v, vs.pdist * w ), vs.st_scale), vs.st_offset)
 return {x=ps.x, y = ps.y, z=v.z, w = w }
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

 for k,v in pairs(drawlist) do
  add(sortlist, {key=-v.key,dl_key=k})
 end
 
 ce_heap_sort(sortlist)

 for p in all(sortlist) do
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
   line(v.a.x, v.a.y, v.b.x, v.b.y, v.col)
end   



function gfx_line( a, b, col )
  line(a.x, a.y, b.x, b.y, col)
end

function gfx_point( a, col )
  pset(a.x, b.x, col)
end


function gfx_3d_line( a, b, col )
  local ac = rt_apply( a, vs.world_to_cam )
  local bc = rt_apply( b, vs.world_to_cam )

  if ac.z < vs.near then
   if bc.z < vs.near then
    return
   else
    -- clip a
    atob = v3_sub(bc, ac)
    t = (vs.near - ac.z) / atob.z
    ac = v3_add( ac, v3_mul_s( atob, t ) )
   end
  elseif bc.z < vs.near then
    -- clip b
    atob = v3_sub(bc, ac)
    t = (vs.near - ac.z) / atob.z
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
   
  local c2 = v3(c.x, c.y, c.z)
  c2.x += p 
  local cp2 = vs_view_to_screen( c2 )

  if ( p > 0 ) circ( cp.x, cp.y, cp2.x - cp.x, col )
end 

function gfx_3d_sprite( c, w, h, sx, sy, sw, sh )
 --if (not vs_cull_sphere( c, max(w,h) )) then return end

  c = rt_apply( c, vs.world_to_cam )
  if (c.z < vs.near ) return

  d = v3_length(c)

  px = get_projected_size( w, d )
  py = get_projected_size( h, d )
   
  c1 = v3_sub( c, v3( px, py, 0) )
  c2 = v3_add( c, v3( px, py, 0) )
  cp1 = vs_view_to_screen( c1 )
  cp2 = vs_view_to_screen( c2 )

  sspr( sx, sy, sw, sh, cp1.x, cp1.y, cp2.x - cp1.x, cp2.y - cp1.y )
end 


function gfx_tri_wire( a, b, c )
	line(a.x, a.y, b.x, b.y)
	line(b.x, b.y, c.x, c.y)
	line(c.x, c.y, a.x, a.y)
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
 if a.y > b.y then
  a,b = b,a
 end

 if b.y > c.y then
  b,c = c,b
  if a.y > b.y then
   a,b = b,a
  end  
 end
 return a,b,c
end

function gfx_tri_fill( a, b, c, col )

	-- sort vertices
 a,b,c = sort_y_3( a, b, c )

 -- ceil and clip y
 ayc = mid( ceil(a.y), vs.sct, vs.scb )
 byc = mid( ceil(b.y), vs.sct, vs.scb )
 cyc = mid( ceil(c.y), vs.sct, vs.scb )

 -- init edges
	dab_dy_x = (b.x - a.x) / (b.y - a.y)
	dac_dy_x = (c.x - a.x) / (c.y - a.y)
 dbc_dy_x = (c.x - b.x) / (c.y - b.y)  

	ab_x = a.x + (ayc - a.y) * dab_dy_x
	ac_x = a.x + (ayc - a.y) * dac_dy_x
 bc_x = b.x + (byc - b.y) * dbc_dy_x

 byc-=1

	for py=ayc,byc do
		line( ab_x, py, ac_x, py, col )
		ab_x += dab_dy_x
		ac_x += dac_dy_x
	end

 byc+=1
 cyc-=1

	for py=byc,cyc do
  line( bc_x, py, ac_x, py, col )
		bc_x += dbc_dy_x
		ac_x += dac_dy_x
	end
end

function hline_tex( xl, xr, y, tex )
 --if y < vs.sc.tl.y or y >= vs.sc.br.y then return end

 if (xr.x < xl.x) xl,xr = xr,xl
 xlf = mid( flr(xl.x), vs.scl, vs.scr )
 xrf = mid( flr(xr.x), vs.scl, vs.scr )

 --x = xlf
 --w = xrf - xlf

 rdx = 1. / (xr.x - xl.x)

 dx_u = (xr.u - xl.u) * rdx
 dx_v = (xr.v - xl.v) * rdx
 dx_w = (xr.w - xl.w) * rdx

 ix = xlf - xl.x
 u = xl.u + ix * dx_u
 v = xl.v + ix * dx_v
 w = xl.w + ix * dx_w

 for x = xlf, xrf do
  rw = 1 / w
  --tu = ((u*rw) % tex.w) + tex.x
  --tv = ((v*rw) % tex.h) + tex.y             
  tu = u*rw + tex.x
  tv = v*rw + tex.y             
     pset(x,y, sget( tu, tv ) )
 u += dx_u
 v += dx_v
 w += dx_w

 end
 --hspan( xl, y, xr-xl, cc )

end

function gfx_tri_tex( a, b, c, tex )
 -- sort vertices
 a,b,c = sort_y_3( a, b, c )

 ayc = mid( ceil(a.y), vs.sct, vs.scb )
 byc = mid( ceil(b.y), vs.sct, vs.scb )
 cyc = mid( ceil(c.y), vs.sct, vs.scb )

 dab_dy_ry = 1. / (b.y - a.y)
 dab_dy_x = (b.x - a.x) * dab_dy_ry
 dab_dy_u = (b.u - a.u) * dab_dy_ry
 dab_dy_v = (b.v - a.v) * dab_dy_ry
 dab_dy_w = (b.w - a.w) * dab_dy_ry

 dac_dy_ry = 1. / (c.y - a.y)
 dac_dy_x = (c.x - a.x) * dac_dy_ry
 dac_dy_u = (c.u - a.u) * dac_dy_ry
 dac_dy_v = (c.v - a.v) * dac_dy_ry
 dac_dy_w = (c.w - a.w) * dac_dy_ry

 dbc_dy_ry = 1. / (c.y - b.y)
 dbc_dy_x = (c.x - b.x) * dbc_dy_ry  
 dbc_dy_u = (c.u - b.u) * dbc_dy_ry  
 dbc_dy_v = (c.v - b.v) * dbc_dy_ry  
 dbc_dy_w = (c.w - b.w) * dbc_dy_ry  

 ab = {}
 ab.x = a.x + (ayc - a.y) * dab_dy_x
 ab.u = a.u + (ayc - a.y) * dab_dy_u
 ab.v = a.v + (ayc - a.y) * dab_dy_v
 ab.w = a.w + (ayc - a.y) * dab_dy_w

 ac = {}
 ac.x = a.x + (ayc - a.y) * dac_dy_x
 ac.u = a.u + (ayc - a.y) * dac_dy_u
 ac.v = a.v + (ayc - a.y) * dac_dy_v
 ac.w = a.w + (ayc - a.y) * dac_dy_w

 bc = {}
 bc.x = b.x + (byc - b.y) * dbc_dy_x
 bc.u = b.u + (byc - b.y) * dbc_dy_u
 bc.v = b.v + (byc - b.y) * dbc_dy_v
 bc.w = b.w + (byc - b.y) * dbc_dy_w

 for py=ayc,byc-1 do
  hline_tex( ab, ac, py, tex )
  ab.x += dab_dy_x
  ab.u += dab_dy_u
  ab.v += dab_dy_v
  ab.w += dab_dy_w

  ac.x += dac_dy_x
  ac.u += dac_dy_u
  ac.v += dac_dy_v
  ac.w += dac_dy_w
 end 

 for py=byc,cyc-1 do
  hline_tex( bc, ac, py, tex )
  bc.x += dbc_dy_x
  bc.u += dbc_dy_u
  bc.v += dbc_dy_v
  bc.w += dbc_dy_w

  ac.x += dac_dy_x
  ac.u += dac_dy_u
  ac.v += dac_dy_v
  ac.w += dac_dy_w
 end
end


function edge_func(a, b, c)
    return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x)
end

function gfx_tri_bary( a, b, c, tex )
   // bounding box
   minx = min3(a.x, b.x, c.x)
   miny = min3(a.y, b.y, c.y)
   maxx = max3(a.x, b.x, c.x)
   maxy = max3(a.y, b.y, c.y)

   minx = max(minx, 0)
   miny = max(miny, 0)
   maxx = min(maxx, 127)
   maxy = min(maxy, 127)

   p = v2(minx,miny)
   b0_row = edge_func(b, c, p)
   b1_row = edge_func(c, a, p)
   b2_row = edge_func(a, b, p)

   btot = b0_row + b1_row + b2_row

   if ( btot <= 0 ) then return end

   rb = 1 / btot

   b0_row *= rb
   b1_row *= rb
   b2_row *= rb

   db0_dx = (b.y - c.y) * rb
   db1_dx = (c.y - a.y) * rb
   db2_dx = (a.y - b.y) * rb

   db0_dy = (c.x - b.x) * rb
   db1_dy = (a.x - c.x) * rb
   db2_dy = (b.x - a.x) * rb

   u_row = b0_row * a.u + b1_row * b.u + b2_row * c.u
   v_row = b0_row * a.v + b1_row * b.v + b2_row * c.v
   w_row = b0_row * a.w + b1_row * b.w + b2_row * c.w

   du_dx = db0_dx * a.u + db1_dx * b.u + db2_dx * c.u
   dv_dx = db0_dx * a.v + db1_dx * b.v + db2_dx * c.v
   dw_dx = db0_dx * a.w + db1_dx * b.w + db2_dx * c.w

   du_dy = db0_dy * a.u + db1_dy * b.u + db2_dy * c.u
   dv_dy = db0_dy * a.v + db1_dy * b.v + db2_dy * c.v
   dw_dy = db0_dy * a.w + db1_dy * b.w + db2_dy * c.w

	for y = miny,maxy do
        b0 = b0_row
        b1 = b1_row
        b2 = b2_row
        u = u_row
        v = v_row
        w = w_row
        for x = minx,maxx do
            if (b0 >= 0 and b1 >= 0 and b2 >= 0) then
            	rw = 1 / w
            	tu = ((u*rw) % tex.w) + tex.x
            	tv = ((v*rw) % tex.h) + tex.y            	
                pset(x,y, sget( tu, tv ) )

                --pset(x,y, 15 )
            end

            b0 += db0_dx
            b1 += db1_dx
            b2 += db2_dx            

	        u += du_dx
    	    v += dv_dx
    	    w += dw_dx
		end

        b0_row += db0_dy
        b1_row += db1_dy
        b2_row += db2_dy

        u_row += du_dy
	    v_row += dv_dy        
	    w_row += dw_dy        
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

dthi = 0

function gfx_dither( grad, s )
   local gc = #grad
   local g = s * (gc-1)
   local g_i = flr(g)   
   local g_f = g - g_i

   local g_i1 = mid( g_i + 1, 1, gc )
   local g_i2 = mid( g_i + 2, 1, gc )

   local c1 = grad[g_i1]
   local c2 = grad[g_i2]

   --pat = grad[flr(s * #grad) + 1]
   local dc = #dither_patterns
   local fp = dither_patterns[ mid( 1 + flr( g_f * (dc-1) ), 1, dc ) ]

   return c1, c2, fp
end


obj_cube = {}
sys_time = { t=0, dt=1/60 }

function _init()
	obj_cube = obj_make_cube()

end

function update(dt)
 sys_time.dt = dt
 sys_time.t += dt
end


function _update60()
 update(1/60)
end

gradients = { 
 {0x0, 0x2, 0x4, 0x9, 0xa, 0x7},  --yellow
 {0x0, 0x1, 0x3, 0xb, 0x7}, -- green
 {0x0, 0x1, 0xc, 0x7}, -- blue
 {0x0, 0x2, 0x8, 0xe, 0x7}  -- red
}

function perf_draw()
 clip()
 local cpu=flr(stat(1)*100)
 local fps=60/ceil(stat(1))
 local perf=
  cpu .. "% cpu @ " ..
  fps ..  " fps"
 print(perf,0,122,0)
 print(perf,0,121,fps==60 and 7 or 8)
end

function obj_calc_normals()
	for t in all(obj.tri) do
		ova = obj.vtx[t[1]]
		ovb = obj.vtx[t[2]]
		ovc = obj.vtx[t[3]]

		dab = v3_sub( ovb, ova )
		dbc = v3_sub( ovc, ovb )

		t.n = v3_normalize( v3_cross( dab, dbc ) )	
	end	
end

function obj_calc_bounds(obj)
	obj.bounds = {}

	fst = 1

	for v in all(obj.vtx) do
		if fst == 1 then
			vmin = v3(v.x, v.y, v.z)
			vmax = v3(v.x, v.y, v.z)
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
	obj.vtx = {}

	obj.vtx[1] = v3(-1, -1, -1)
	obj.vtx[1].u = 0
	obj.vtx[1].v = 0

	obj.vtx[2] = v3(1, -1, -1)
	obj.vtx[2].u = 16
	obj.vtx[2].v = 0

	obj.vtx[3] = v3(-1, 1, -1)
	obj.vtx[3].u = 0
	obj.vtx[3].v = 16

	obj.vtx[4] = v3(1, 1, -1)
	obj.vtx[4].u = 16
	obj.vtx[4].v = 16


	obj.vtx[5] = v3(1, -1, 1)
	obj.vtx[5].u = 0
	obj.vtx[5].v = 0

	obj.vtx[6] = v3(-1, -1, 1)
	obj.vtx[6].u = 16
	obj.vtx[6].v = 0

	obj.vtx[7] = v3(1, 1, 1)
	obj.vtx[7].u = 0
	obj.vtx[7].v = 16

	obj.vtx[8] = v3(-1, 1, 1)
	obj.vtx[8].u = 16
	obj.vtx[8].v = 16

	tex = { x = 8, y = 0, w = 16, h = 16 }

	obj.tri = { 
		{ 1, 2, 3, t=nil, c = 1 },
		{ 2, 4, 3, t=nil, c = 1 },
		{ 5, 6, 7, t=nil, c = 1 },
		{ 6, 8, 7, t=nil, c = 1 },
		{ 6, 1, 8, t=nil, c = 2 },
		{ 8, 1, 3, t=nil, c = 2 },
		{ 2, 5, 7, t=nil, c = 2 },
		{ 2, 7, 4, t=nil, c = 2 },
		{ 6, 5, 1, t=nil, c = 3 },
		{ 5, 2, 1, t=nil, c = 3 },
		{ 3, 7, 8, t=nil, c = 4 },
		{ 3, 4, 7, t=nil, c = 4 },
	}

 obj.line = {
  { 1, 2, c = 7 },
  { 2, 4, c = 7 },
  { 4, 3, c = 7 },
  { 3, 1, c = 7 }
 }
	
	obj_finalize(obj)

	return obj
end

function transform_vert( ov )
   local vv = rt_apply( ov, vs.obj_to_cam )

   --local vw = rt_apply( ov, vs.obj_to_world )   
   --local vv = rt_apply( vw, vs.world_to_cam )
   vv = vs_view_to_screen( vv )
   
   -- vv.u = ov.u * vv.w -- todo: remove if untextured
   -- vv.v = ov.v * vv.w -- todo: remove if untextured
   return vv
end

function transform_vert_shadow( ov )
   --local vv = rt_apply( ov, vs.obj_to_cam )

   local vw = rt_apply( ov, vs.obj_to_world )   
   vw.x += vw.y
   vw.y = 0
   local vv = rt_apply( vw, vs.world_to_cam )
   vv = vs_view_to_screen( vv )
   
   -- vv.u = ov.u * vv.w -- todo: remove if untextured
   -- vv.v = ov.v * vv.w -- todo: remove if untextured
   return vv
end

function obj_draw( obj, obj_to_world, shadow )
 vs_set_obj_mat( obj_to_world )

	scr_vtx = {}

 if not shadow then
 	for i,ov in pairs(obj.vtx) do
   scr_vtx[i] = transform_vert(ov)
 	end

  ldir = v3(0,1,0)
  obj_ldir = v3_mul_m3( ldir, vs.world_to_obj_rot )

 else
  for i,ov in pairs(obj.vtx) do
   scr_vtx[i] = transform_vert_shadow(ov)
  end   -- else code
 end

 --for sv in all(scr_vtx) do
  --circ( sv.x, sv.y, 1.5, 4 )
 --end


	for t in all(obj.tri) do

 a = scr_vtx[t[1]]
 b = scr_vtx[t[2]]
 c = scr_vtx[t[3]]

 -- backface cull
 if v2_cross( v2_sub( b, a ), v2_sub( c, b ) ) < 0.0 then
  if a.z > vs.near and b.z > vs.near and c.z > vs.near then

  local col, fp

 if shadow then
  fillp(0b0101101001011010.1)
  gfx_tri_fill( a, b, c, 0 )

 else
   ldotn = v3_dot(obj_ldir, t.n)
   s = sat( ldotn * -0.5 + 0.5)

   local c1, c2
   c1,c2,fp = gfx_dither( gradients[t.c], s )   
   col = c1 + c2 * 16  

    local key = (a.z + b.z + c.z) / 3
    add( drawlist, { key=key, fn = dl_tri, value = {a=a, b=b, c=c, col=col, fp=fp } } )
 end

     --fillp( fp )
     --gfx_tri_tex( a, b, c, t.t )
     --gfx_tri_fill( a, b, c, col )         		
  		--gfx_tri_bary( scr_vtx[t[1]], scr_vtx[t[2]], scr_vtx[t[3]], t.t )     
  		--gfx_tri_wire( scr_vtx[t[1]], scr_vtx[t[2]], scr_vtx[t[3]] )
    end
   end
  end

 for l in all(obj.line) do
  local a = scr_vtx[l[1]]
  local b = scr_vtx[l[2]]
  if a.z > vs.near and b.z > vs.near then
   if shadow then
    fillp(0b0101101001011010.1)
    line(a.x, a.y, b.x, b.y, 0)
   else
    local key = (a.z + b.z) / 2
    add( drawlist, { key=key, fn = dl_line, value = {a=a, b=b, col=l.c } } )
   end
  end
 end

 dl_draw()
 dl_reset()


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
    if item.bg then item.draw(item.value, bg) end
   end
  end
 else
  sortlist = {}

  for k,v in pairs(scene) do
   add(sortlist, {key=-v.key,sc_key=k})
  end
  
  ce_heap_sort(sortlist)

  for p in all(sortlist) do
   item = scene[p.sc_key]
   if item.fg then item.draw(item.value, bg) end
  end
 end
end

function scene_add_obj( obj, obj_to_world )
 local t = obj_to_world.t
 local rt = {}
 rt.t = v3(t.x, t.y, t.z)
 local r = obj_to_world.r
 rt.r = { {r[1][1], r[1][2], r[1][3]}, {r[2][1], r[2][2], r[2][3]}, {r[3][1], r[3][2], r[3][3]} }

 local bounds_c_world = rt_apply( obj.bounds.c, obj_to_world ) 
 if vs_cull_sphere( bounds_c_world, obj.bounds.r ) then
  vp = v3_sub(bounds_c_world, vs.cam_to_world.t)
  cz = m3_get_az( vs.cam_to_world.r )
  key = -v3_dot( vp, cz )
  add( scene,
  { 
   key = key, 
   draw = scene_draw_obj,
   bg = true,
   fg = true,   
   value = {
    obj = obj,
    rt = rt
   } 
  } )
 end
end

function scene_build()

 scene_reset()

 add( scene,
 { 
  key = -32767, 
  draw = scene_draw_background,
  bg = true,
  fg = false,
  value = {} 
 } )

 local y_rot = sys_time.t * 1
 local x_rot = sys_time.t * 0.234

 local obj_r1 = m3_rot_y(y_rot)
 local obj_r2 = m3_rot_x(x_rot)

 local obj_to_world =
  { r=m3_mul( obj_r2, obj_r1 ), 
    t=v3( 0, 2, 0 ) }

   --gfx_3d_sphere_outline( rt_apply( cube.bounds.c, obj_to_world ), cube.bounds.r )
   -- gfx_3d_sprite( rt_apply( cube.bounds.c, obj_to_world ), cube.bounds.r, cube.bounds.r * 0.75, 8, 0, 16, 16 )
   
   --for z=10,-10,-1 do
    --gfx_3d_sprite( v3(4,1,z), 0.5, 1, 40, 0, 8, 16 )
    --gfx_3d_sprite( v3(-4,1,z), 0.5, 1, 40, 0, 8, 16 )
   --end

   scene_add_obj( obj_cube, obj_to_world )
 
   for x=2,5 do
     obj_to_world.t.x = x * 4
     scene_add_obj( obj_cube, obj_to_world )
   end    


end

function vgrad(y0, y1, i0, i1, g)
 local s = i0
 local ds_dy = (i1 - i0) / (y1 - y0)
 for y=y0,y1 do
  c1,c2,fp = gfx_dither( g, s )   
  fillp(fp)
  line(0,y, 127,y, c1 + c2 * 16)

  s += ds_dy
 end
end

function scene_draw_background()
  draw_floor()
  fillp()

  gfx_3d_grid(6)
end

function scene_draw_obj( scene_obj, bg )
 obj_draw( scene_obj.obj, scene_obj.rt, bg )
end

function draw_floor()
 local d = 1000
 local za = vs.cam_to_world.r[3]
 local z = v3(za[1],za[2],za[3])
 z.y = 0
 z = v3_normalize(z)
 pw = v3_add(v3_mul_s(z,d), vs.cam_to_world.t)
 pw.y = 0
 
 local vv = rt_apply( pw, vs.world_to_cam )
 vv = vs_view_to_screen( vv )

 y = vv.y
 if ( y <= 127 ) rectfill(0,0,127,y,1)
 if ( y > 0 ) rectfill(0,y,127,127,3)
 --vgrad(0,63, 0.6 + light * 0.3, 0.1 + light * 0.05 , gradients[1] )
 --vgrad(64, 127, 0.1 + light * 0.05, 0.6+ light * 0.2, gradients[2] )
end

cam_pos = v3(0,1,-10)
cam_angles = v3(0,0,0)

function _draw()

	cls()

	--map(0,0, 0,0, 16,16)

	-- gfx_tri_fill( v2(10, 30), v2(50, 40), v2(25, 100), 0x1111 )

-- drawlist

dl_reset()


   cam_move = v3(0,0,0)
      if (btn(4)) then
       if ( btn(0))cam_move.x-=.1
       if ( btn(1))cam_move.x+=.1

       if ( btn(2))cam_move.z+=.1
       if ( btn(3))cam_move.z-=.1
      else
       if ( btn(0))cam_angles.y-=.01
       if ( btn(1))cam_angles.y+=.01

       if ( btn(2))cam_angles.x+=.01
       if ( btn(3))cam_angles.x-=.01
      end

   cam_m = m3_mul( m3_rot_y(cam_angles.y), m3_rot_x(cam_angles.x) )

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

   --gfx_3d_line( v3(0,0,0), v3(3, 0, 0), 4)

end

	perf_draw()
end

__gfx__
0000000099aaaaaaaaa777aab3b33333444444445666667600000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000999999999999999ab3b333b3494424945555567600000000000000000000000000000000000000000000000000000000000000000000000000000000
0070070099a444222220209733333333444444440566776000000000000000000000000000000000000000000000000000000000000000000000000000000000
0007700099a99a499a4992973b333333424444440556676000000000000000000000000000000000000000000000000000000000000000000000000000000000
0007700049a99a499a4990973b33b3b3444942440566776000000000000000000000000000000000000000000000000000000000000000000000000000000000
0070070049a99a499a49909a33333333444444440556676000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000029799a499a49909a33b33b33942444240566676000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000029799a499a49929a33333b33444444440556776000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000049a99a499a49929a00000000000000000566676000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000049a99a499a49929a00000000000000000556676000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000049a99a499a49949a00000000000000000566776000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000029799a499a49949a00000000000000000556676000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000029a99a499a49949900000000000000000566676000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000029aa7aaaa7aa7a9900000000000000000556676000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000499999999999999900000000000000005667677700000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000444222244449999900000000000000005556667600000000000000000000000000000000000000000000000000000000000000000000000000000000
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

