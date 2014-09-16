

nseed = 2;

for i=1:nseed
    %Set the seed
    s = RandStream('mcg16807','Seed',i*1000) ; RandStream.setGlobalStream(s);
    
end


