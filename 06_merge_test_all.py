import sys
import os
join = os.path.join
import shutil

def run():
    path_to_out = "out_"+sys.argv[0].split(".")[0]

    num_portia = 208
    path_to_portia = "out_02_test"
    
    num_cato = 208
    path_to_cato = "out_cato/out_02_test"

    num_deltow = 287
    path_to_deltow = "out_deltow/out_02_test"

    if(not os.path.exists(path_to_out)):
        os.makedirs(path_to_out)

    n = 0
    for i in range(num_portia):
        dname1 = "{0:03d}".format(i)
        dname2 = "{0:03d}".format(n)
        n+=1
        shutil.move(join(path_to_portia, dname1),
                        join(path_to_out,  dname2))
        
    for i in range(num_cato):
        dname1 = "{0:03d}".format(i)
        dname2 = "{0:03d}".format(n)
        n+=1        
        shutil.move(join(path_to_cato, dname1),
                        join(path_to_out,  dname2))

    for i in range(num_deltow):
        dname1 = "{0:03d}".format(i)
        dname2 = "{0:03d}".format(n)
        n+=1        
        shutil.move(join(path_to_deltow, dname1),
                        join(path_to_out,  dname2))        
        
if __name__=="__main__":
    run()
    
