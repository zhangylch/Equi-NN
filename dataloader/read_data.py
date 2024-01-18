import numpy as np

# read system configuration and energy/force
def Read_data(datafloder="train/",force_table=None,Dtype=np.float32):
    coor=[]
    cell=[]
    pot=[] 
    force=None
    species=[]
    numatoms=[]
    mass=[]
    #===================variable for force====================
    if force_table==1:
       force=[]
    numpoint=0
    fname2=datafloder+'configuration'
    with open(fname2,'r') as f1:
        while True:
            string=f1.readline()
            if not string: break
            icell=np.zeros((3,3),dtype=Dtype,order="F")
            numatom=int(string)
            numatoms.append(numatom)
            string=f1.readline()
            # here to save the coordinate with row first to match the neighluist in fortran
            m=np.array(list(map(float,string.split())))
            icell[:,0]=m
            string=f1.readline()
            m=np.array(list(map(float,string.split())))
            icell[:,1]=m
            string=f1.readline()
            m=np.array(list(map(float,string.split())))
            icell[:,2]=m
            icoor=np.zeros((3,numatom),dtype=Dtype,order="F")
            imass=np.zeros(numatom,dtype=Dtype)
            ispecies=np.zeros((numatom,1),dtype=np.int32)
            if force_table==1: iforce=np.zeros((numatom,3),dtype=Dtype)
            for num in range(numatom):
                string=f1.readline()
                m=string.split()
                tmp=np.array(list(map(float,m[1:])))
                imass[num]=tmp[0]
                ispecies[num,0]=tmp[1]
                icoor[:,num]=tmp[2:5]
                if force_table: iforce[num]=tmp[5:8]
            string=f1.readline()
            pot.append(float(string.split()[1]))
            numpoint+=1
            coor.append(icoor)
            mass.append(imass)
            cell.append(icell)
            species.append(ispecies)
            if force_table: force.append(iforce)
    return numpoint,coor,mass,cell,species,numatoms,pot,force
