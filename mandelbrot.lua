--Mandelbrot set image explorer by Phillip Alvelda 
RMode = "Color"
WIDTH = 50
HEIGHT = 50
tileNumber = 150
tileSize = WIDTH/tileNumber
tilesize= tileSize

MinRe = -2.0
MaxRe = 2
MinIm = -2.0
MaxIm = 2
Re_factor = (MaxRe-MinRe)/(WIDTH -1)
Im_factor = (MaxIm-MinIm)/(HEIGHT -1)
MaxIterations = 255
UCounter =0
count=0
red={}
green={}
blue ={}
tiles = {}
for y= 1,HEIGHT/tilesize do
	tiles[y] = {}
end

-- Use this function to perform your initial setup
function setup() 
	setInstructionLimit(0)
	InitColorMap()
	ManSetCalc()
end

function InitColorMap()
	for i=0,MaxIterations do
		if RMode == "Iterations" then
			red[i]=255-i*255/MaxIterations
			green[i]=255-i*255/MaxIterations
			blue[i]=255-i*255/MaxIterations
		elseif RMode == "Color" then
			red[i]=(i*8) % 255
			green[i] = (i*9) % 255
			blue[i] = (i * 4) % 255
		elseif RMode == "Binary" then
			if (value % 2) == 0 then
				red[i]=255
				green[i]=255
				blue[i]=255  
			else 
				red[i]=0
				green[i]=0
				blue[i]=0  
			end
		end         
	end    
end



function touched(touch)
	if touch.state == ENDED then --only use the final touch event to scale image window
		dre=MaxRe-MinRe
		dim=MaxIm-MinIm
		x = MinRe + touch.x/WIDTH * dre
		y = MinIm + touch.y/HEIGHT * dim
		print ("touch x=",x,"y=",y)

		--set the zoom factor and new window borders
		MinRe = x -  dre / 3
		MaxRe = x + dre / 3
		MinIm = y - dim / 3
		MaxIm = y + dim / 3
		Re_factor = (MaxRe-MinRe)/(WIDTH -1)
		Im_factor = (MaxIm-MinIm)/(HEIGHT -1)
		print "Zoom in, Regen..."
		ManSetCalc()
		print "Done."
	end
end

function ManSetCalc() 
	print("Calc Start",ElapsedTime )    
	local y, x, n, isInside
	local ylim=HEIGHT/tilesize
	local xlim=WIDTH/tilesize
	local ltiles=tiles
	local imf=Im_factor*tilesize
	local ref=Re_factor*tilesize
	local iterations
	for y = 1, ylim do
		c_im = MinIm + y*imf

		for x = 1, xlim do
			c_re = MinRe + x*ref
			Z_re = c_re
			Z_im = c_im
			isInside = true
			for n=0, MaxIterations, 1 do
				Z_re2 = Z_re*Z_re
				Z_im2 = Z_im*Z_im
				iterations=n
				if(Z_re2 + Z_im2 > 4)then
					isInside = false   
					break
				end
				Z_im = 2*Z_re*Z_im + c_im
				Z_re = Z_re2 - Z_im2 + c_re
			end            
			ltiles[y][x] = iterations                 
		end
	end 
	print("End",ElapsedTime)  
end

function draw()
	print("Draw interval=",DeltaTime)
	count = count + 1
	noSmooth()
	background(0, 0, 0, 255)
	scale(tilesize,tilesize)
	local lred=red
	local lgreen=green
	local lblue=blue
	local ltiles=tiles
	local x,y,column, value
	local lrect=rect
	local lfill=fill
	local lipairs=ipairs
	for y,column in lipairs(ltiles) do    
		for x,value in lipairs(column) do
			lfill(lred[value],lgreen[value],lblue[value],255)
			--fill((value*8) % 255, (value*9) % 255,(value*4) % 255,255)
			lrect(x, y, 1, 1)
		end   
	end
end
