"""Exact symmetric collision of two planar Euler shocks, independent of detection.

Before meeting, two shocks propagate into resting uniform gas. At the meeting
point, their downstream streams form a symmetric Riemann problem with two
outgoing shocks and a resting central state. No contact has nonzero strength
in this symmetric problem. Units: box length=1, ambient density=pressure=1.
"""
from dataclasses import dataclass

import numpy as np


GAMMA=5/3


def conserved(primitive,gamma=GAMMA):
    """Convert (..., rho/u/P) arrays to (..., rho/momentum/energy)."""
    q=np.asarray(primitive,float)
    rho,u,p=np.moveaxis(q,-1,0)
    return np.stack([rho,rho*u,p/(gamma-1)+.5*rho*u*u],axis=-1)


def primitive(conserved_state,gamma=GAMMA):
    q=np.asarray(conserved_state,float)
    rho,momentum,energy=np.moveaxis(q,-1,0)
    u=momentum/rho
    return np.stack([rho,u,(gamma-1)*(energy-.5*rho*u*u)],axis=-1)


def flux(primitive_state,gamma=GAMMA):
    rho,u,p=np.asarray(primitive_state,float)
    energy=p/(gamma-1)+.5*rho*u*u
    return np.array([rho*u,rho*u*u+p,u*(energy+p)])


def shock_ratios(mach,gamma=GAMMA):
    if not np.isfinite(mach) or mach<=1 or gamma<=1:raise ValueError('Require Mach>1 and gamma>1.')
    m2=mach*mach
    return (gamma+1)*m2/((gamma-1)*m2+2),(2*gamma*m2-(gamma-1))/(gamma+1)


@dataclass(frozen=True)
class SymmetricShockCollision:
    incoming_mach: float=3.
    initial_half_separation: float=.25
    gamma: float=GAMMA

    def __post_init__(self):
        if self.gamma!=GAMMA:raise ValueError('Production finder uses gamma=5/3; keep this benchmark consistent.')
        shock_ratios(self.incoming_mach,self.gamma)
        if not 0<self.initial_half_separation<.5:raise ValueError('Initial shocks must lie inside the box.')

    @property
    def ambient(self):return np.array([1.,0.,1.])

    @property
    def incoming_speed(self):return self.incoming_mach*np.sqrt(self.gamma)

    @property
    def collision_time(self):return self.initial_half_separation/self.incoming_speed

    @property
    def incoming_left(self):
        r,p=shock_ratios(self.incoming_mach,self.gamma)
        return np.array([r,self.incoming_speed*(1-1/r),p])

    @property
    def incoming_right(self):return self.incoming_left*np.array([1.,-1.,1.])

    @property
    def outgoing_mach(self):
        rho,u,p=self.incoming_left
        mu=u/np.sqrt(self.gamma*p/rho)
        a=(self.gamma+1)*mu/2
        # u/c = 2/(gamma+1) * (M - 1/M), from mass and momentum RH.
        return .5*(a+np.hypot(a,2.))

    @property
    def outgoing_speed(self):
        rho,u,p=self.incoming_left
        return self.outgoing_mach*np.sqrt(self.gamma*p/rho)-u

    @property
    def central(self):
        r,p=shock_ratios(self.outgoing_mach,self.gamma)
        return np.array([self.incoming_left[0]*r,0.,self.incoming_left[2]*p])

    def structure(self,time):
        if not np.isfinite(time) or time<0:raise ValueError('Time must be finite and nonnegative.')
        if time<self.collision_time:
            half=self.initial_half_separation-self.incoming_speed*time
            middle=self.ambient;phase='incoming';mach=self.incoming_mach
            speeds=[self.incoming_speed,-self.incoming_speed];directions=[1.,-1.]
        elif time>self.collision_time:
            half=self.outgoing_speed*(time-self.collision_time)
            middle=self.central;phase='outgoing';mach=self.outgoing_mach
            speeds=[-self.outgoing_speed,self.outgoing_speed];directions=[-1.,1.]
        else:
            half=0.;middle=self.central;phase='meeting';mach=None
            speeds=[];directions=[]
        if half>=.5:raise ValueError('Reference shocks have reached the box boundary.')
        return dict(phase=phase,positions=np.array([.5-half,.5+half]),
            states=np.stack([self.incoming_left,middle,self.incoming_right]),
            mach=mach,speeds=np.array(speeds),hot_to_cold_directions=np.array(directions),
            two_distinct_fronts=half>0)

    def time_at_separation(self,phase,separation_cells,resolution):
        if not np.isfinite(separation_cells) or separation_cells<0 or resolution<16:
            raise ValueError('Require nonnegative separation and at least 16 cells.')
        half=separation_cells/(2*resolution)
        if phase=='incoming':time=self.collision_time-half/self.incoming_speed
        elif phase=='outgoing':time=self.collision_time+half/self.outgoing_speed
        else:raise ValueError('Phase must be incoming or outgoing.')
        if time<0:raise ValueError('Requested incoming separation precedes the initial state.')
        return time

    def cell_averages(self,resolution,time,filter_cells=1):
        structure=self.structure(time)
        edges=np.arange(resolution+1,dtype=float)/resolution
        left,right=structure['positions'];dx=1/resolution
        a=np.clip((left-edges[:-1])/dx,0,1)
        c=np.clip((edges[1:]-right)/dx,0,1)
        weights=np.stack([a,np.maximum(1-a-c,0),c],axis=1)
        values=weights@conserved(structure['states'],self.gamma)
        return primitive(box_filter(values,filter_cells),self.gamma),structure

    def flux_residuals(self):
        """Independent conservation residuals at all four reference shock branches."""
        branches=[(self.incoming_left,self.ambient,self.incoming_speed),
                  (self.ambient,self.incoming_right,-self.incoming_speed),
                  (self.incoming_left,self.central,-self.outgoing_speed),
                  (self.central,self.incoming_right,self.outgoing_speed)]
        results=[]
        for left,right,speed in branches:
            a,b=conserved(left,self.gamma),conserved(right,self.gamma)
            fa,fb=flux(left,self.gamma),flux(right,self.gamma)
            scale=np.maximum.reduce([abs(fa),abs(fb),abs(speed*a),abs(speed*b),np.ones(3)])
            results.append((fb-fa-speed*(b-a))/scale)
        return np.asarray(results)


def box_filter(values,width):
    if not isinstance(width,(int,np.integer)) or width<1 or width%2!=1:
        raise ValueError('Filter width must be a positive odd number of cells.')
    if width==1:return np.asarray(values,float).copy()
    padded=np.pad(values,((width//2,width//2),(0,0)),mode='edge')
    return np.stack([np.convolve(padded[:,i],np.ones(width)/width,mode='valid') for i in range(3)],axis=1)


def isolated_control(reference,resolution,phase,filter_cells=1):
    """One shock with the same plateau states as a collision branch."""
    if phase=='incoming':
        states=np.stack([reference.incoming_left,reference.ambient])
        mach=reference.incoming_mach;direction=1.
    elif phase=='outgoing':
        states=np.stack([reference.incoming_left,reference.central])
        mach=reference.outgoing_mach;direction=-1.
    else:raise ValueError('Unknown isolated-shock phase.')
    edges=np.arange(resolution+1)/resolution
    fraction=np.clip((.5-edges[:-1])*resolution,0,1)
    average=np.stack([fraction,1-fraction],axis=1)@conserved(states,reference.gamma)
    return primitive(box_filter(average,filter_cells),reference.gamma),dict(
        phase='isolated_'+phase,positions=np.array([.5]),mach=mach,
        hot_to_cold_directions=np.array([direction]),two_distinct_fronts=False)


def contact_control(resolution,filter_cells=1):
    """Stationary isobaric contact: different rho/T, equal velocity and pressure."""
    states=np.array([[1.,0.,1.],[4.,0.,1.]])
    edges=np.arange(resolution+1)/resolution
    f=np.clip((.5-edges[:-1])*resolution,0,1)
    average=np.stack([f,1-f],axis=1)@conserved(states)
    return primitive(box_filter(average,filter_cells)),dict(
        phase='contact',positions=np.array([]),mach=None,hot_to_cold_directions=np.array([]),two_distinct_fronts=False)
