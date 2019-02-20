# Evolutionary Algorithm for PSF Optimization 
# Citation as Engineering and Optimization of Quasi-nondiffracting Helicon-like Beams with an Evolutionary Algorithm 
# Bryce Schroeder, Zhen Zhu, Changliang Guo and Shu Jia, IEEE Photonics 9, 6101109 (2017)
# (C) 2016 
# by Bryce Schroeder, for Shu Jia's lab, Stony Brook University
# sites.google.com/site/thejialab/   bryce.schroeder@gmail.com   www.bryce.pw/
import numpy as np
import matplotlib
import scipy.optimize as sopt


import random
import bisect
import copy

# We define our organism in an object-oriented fashion, first with the 
# general case, and then with a subclass specific to the helicon PSF 
# which we are optimizing. 
class Organism(object):
    '''Represents an individual in the population.
       Subclasses need to define breed_from, random_mutation and generate_random.'''
    serial_number = 0
    valid = False
    MUTATION_CHANCE = 40 # Mutation chance for new organism. 
    def __init__(self, parents=None):
        Organism.serial_number += 1
        self.valid = False
        self.score = None
        self.serial_number = Organism.serial_number
        '''Generate a new organism. If a list of other organisms is passed as
           the parameter 'parents', breed the new organism from them. Otherwise,
           generate the new organism de novo.'''
        self.genes = None
        if parents:
            self.breed_from(parents)
            # Organisms can have many mutations, or none; mutations are not the only source
            # of genetic diversity in the population: the breeding process creates new 
            # combinations of gene analogs, and randomly generated solutions can be injected
            # into each generation as well to provide "fresh blood."
            while random.randint(0,100) < self.MUTATION_CHANCE:
                self.random_mutation()
        else:
            self.generate_random()

class ModHeliconOrganism(Organism):
    valid = False
    MAX_PITCH = 10 #
    MIN_PITCH = -10  # 
    score_gof = 1000.0
    score_s = 1000.0
    score_ar = 1000.0
    MUTATION_OFFSET_MAGNITUDE = 5.00 # 5 how big is a phase offset mutation, maximally? +/-
    
    
    def pretty_print_parameters(self):
        '''Return a good visual representation of the parameters.'''
        return ' '.join(["%+.2f"%n_off for n_off in self.parameters])
    
    def generate_random(self):
        '''Generate this organism's parameters _de novo_.'''
        # Generate a list of (mode, offset) tuples, which will be the parameters.
        self.parameters = [random.random()*(self.MAX_PITCH-self.MIN_PITCH)+self.MIN_PITCH for _ in xrange(3)]
        
    def breed_from(self, parents):
        '''Create a new organism with a combination of traits from the parents.'''
        # FIXME: Currently, only asexual reproduction is supported
        self.parameters = copy.copy(random.choice(parents).parameters)
        
    def random_mutation(self):
        '''Make a single mutation to this organism.'''
        which = random.randint(0, len(self.parameters)-1)
        self.parameters[which] += random.random()*self.MUTATION_OFFSET_MAGNITUDE*2 - self.MUTATION_OFFSET_MAGNITUDE
        self.parameters[which] = min(max(self.MIN_PITCH, self.parameters[which]), self.MAX_PITCH)
        return 'ofs'
        
    def phase_pattern(self, fx, fy):
        '''Render the phase modulation pattern to be imposed on the Fourier plane.'''
        b,c,d = self.parameters
        size = 1 #((size+10)/(20))*0.25 + (0.50-0.125)
        x0 = 0.5
        y0 = 0.5
        #print "x0", x0, np.min(fx), np.max(fx)
        #print "y0", y0, np.min(fy), np.max(fy)
        #if not HeliconOrganism.hasattr('distance'): HeliconOrganism.distance = np.sqrt(fx**2 + fy**2)
        r = 2.3*np.sqrt((fx-x0)**2 + (fy-y0)**2)/size
                
        #itch = a*r**3 + b*r**2 + c*r +d
        pitch = b*r**2 + c*r + d
        
        pattern = 0
        pattern += np.exp(1j*pitch*(np.arctan2( (fy-y0),(fx-x0) )))
                
            
        return pattern*(r < size/2.0)
        

MAX_SCORE = 1000.0
# Here we define our generalized FITNESS FUNCTION, which will be used
# to evaluate the fitness of the population
class Scorer(object):
    # The scorer will encapsulate the scoring criteria
    # and associated constants.
    SIZE=30.0
    TARGET_SIZE = 1.0
    #
    # Score weight for the goodness of the gaussian fit as measured by mean square error
    GOF_WEIGHT = 1375.0
    AR_WEIGHT = 5.0
    SIZE_WEIGHT = 0.75
    def __init__(self, resolution=600):
        '''Set up a new scorer. Higher resolutions yield somewhat better 
           results but everything takes longer to compute.'''
        self.resolution = resolution
        
        self.fx, self.fy = np.meshgrid(np.linspace(0, 1, resolution), 
                                       np.linspace(0, 1, resolution))
    def focal_plane_psf(self, pattern):
        return np.fft.ifftshift(np.fft.ifft2(pattern))
    
    def phase_pattern(self, organism):
        return organism.phase_pattern(self.fx, self.fy)
    
    def gaussian_fit(self, intensity):
        def elliptical_gaussian(xy, amplitude, x0, y0, wx, wy, theta):
            x,y = xy
            p1 = np.cos(theta)**2/(2*wx**2) + np.sin(theta)**2/(2*wy**2)
            p2 = np.sin(theta)**2/(2*wx**2) + np.cos(theta)**2/(2*wy**2)
            p3 = -np.sin(2*theta)/(4*wx**2) + np.sin(2*theta)/(4*wy**2)
            result = (amplitude*np.exp(- (  p1*((x-x0)**2) 
                                 + p2*((y-y0)**2)
                                 + 2*p3*(x-x0)*(y-y0))))
            result = np.abs(result.ravel())
            return result

            #return background + amplitude*np.exp( 
            #      (x0*np.cos(theta) + y0*np.sin(theta))**2/wx**2  
            #    - (x0*np.sin(theta) - y0*np.cos(theta))**2/wy**2 
            #    )
        guess = (0.001, 0.5, 0.5, -0.010, -0.010, 0.0)
        try:
            popt, pcov = sopt.curve_fit(elliptical_gaussian, (self.fx,self.fy), 
                                        intensity.ravel(), p0=guess)
        except RuntimeError:
            return None,guess
        
        fitted = elliptical_gaussian((self.fx,self.fy), *popt).reshape(*intensity.shape)
        return fitted, popt
    
    def score(self, organism):
        '''Total score and weighted component scores are returned, as well as images'''
        if organism.score is not None:
            return organism.score, (organism.score_gof, organism.score_ar, organism.score_s), (None,None,None,None)
            
        pattern = self.phase_pattern(organism)
        fp = self.focal_plane_psf(pattern)
        intensity = np.abs(fp**2)
        vmax,vmin = np.max(intensity), np.min(intensity)
        fitted, (amplitude, x0, y0, wx, wy, theta) = self.gaussian_fit(intensity)
        print "**", amplitude, wx, wy
        if fitted is None: return MAX_SCORE, (MAX_SCORE,MAX_SCORE,MAX_SCORE), (pattern, fp, intensity, fitted)
        # Mean square error of the gaussian fit
        nm_fit = (fitted - vmin)/(vmax-vmin)
        nm_in  = (intensity - vmin)/(vmax-vmin)
        score_goodness_of_fit = self.GOF_WEIGHT*np.sum((nm_fit-nm_in)**2)/self.resolution**2
        
        # Score for the aspect ratio
        score_aspect_ratio = self.AR_WEIGHT*((min(np.abs(wx),np.abs(wy))/max(np.abs(wx),np.abs(wy))) - 0.5)**2
        
        # Score for the size
        score_size = self.SIZE_WEIGHT*(max(np.abs(wx),np.abs(wy)) - self.TARGET_SIZE)**2
        
        organism.score = score_goodness_of_fit + score_aspect_ratio + score_size
        organism.score_gof = score_goodness_of_fit
        organism.score_ar = score_aspect_ratio
        organism.score_s = score_size
        
        return organism.score, (
            score_goodness_of_fit, score_aspect_ratio, score_size), (
                pattern, fp, intensity, fitted)
    
    def elitism(self, population, elite_cut, subscore_cut):
        elite = sorted(population, key=lambda o: o.score)[:elite_cut]
        for key in [lambda o: o.score_gof, lambda o: o.score_ar, lambda o: o.score_s]:
            elite.extend(sorted(population, key=key)[:subscore_cut])
        return elite
    
    def statistics(self, population):
        ''' expects a fully scored & validated population '''
        valids = filter(lambda a: a.valid, population)
        valid_fraction = len(valids)/float(len(population))
        scores, scores_gof, scores_ar, scores_s = map(np.asarray, 
                                                      zip(*[(a.score, a.score_gof, a.score_ar, a.score_s) for a in valids]))
        return (valid_fraction, 
                np.min(scores), np.min(scores_gof), np.min(scores_ar), np.min(scores_s),
                np.mean(scores), np.mean(scores_gof), np.mean(scores_ar), np.mean(scores_s),
                )
    




class Environment(object):
    ELITE_CUT = 2 # Take the best 1 organism by total score and copy them unchanged to the next generation.
    SUBSCORE_ELITE_CUT = 1 # take the best organism by each subscore and also copy it unchanged to the next generation.
    DE_NOVO_CUT = 1 # how many random organisms to add each generation. Limited utility until sexual reproduction is
    # implemented.
    
    def __init__(self, OrganismClass, scorer, count=50):
        self.count = count
        self.generation = 0
        self.OrganismClass = OrganismClass
        self.organisms = [OrganismClass() for _ in xrange(count)]
        self.scorer = scorer
        
    def iterate(self, logfile=None):
        self.logfile=logfile
        # From which parents for the next generation will be drawn.
        # as a side-effect of score_valid_population, all the organisms will
        # also be scored now.
        wheel = self.breeding_wheel(*self.score_valid_population())
        
        self.statistics = self.scorer.statistics(self.organisms)
        
        next_generation = self.scorer.elitism(self.organisms, self.ELITE_CUT, self.SUBSCORE_ELITE_CUT)
        while len(next_generation) < self.count - self.DE_NOVO_CUT:
            next_generation.append(self.OrganismClass(parents = [random.choice(wheel)])) # asexual reproduction only, FIXME
            
        self.organisms = next_generation + [self.OrganismClass() for _ in xrange(self.DE_NOVO_CUT)]
        
        
    def score_valid_population(self):
        valids = []
        scores = []
        for organism in self.organisms:
            self.scorer.score(organism)
            if organism.score and organism.score < MAX_SCORE-1:
                if self.logfile:
                    print >> self.logfile, organism.score, organism.score_gof, organism.score_ar, organism.score_s, organism.pretty_print_parameters()

                valids.append(organism)
                organism.valid = True
                scores.append(organism.score)
            else:
                organism.valid = False
        return scores, valids
        
    def breeding_wheel_proportions(self, scores, organisms, wheel_size=100):
        scores = np.asarray(scores)
        
        best_score = np.min(scores)
        worst_score = np.max(scores)
        
        normalized_scores = (scores - best_score)/(worst_score-best_score)
        # Now the best organism has a score of 0 and the worst has a score of 1
        positive_scores = 1.0 - normalized_scores
        total_score = np.sum(positive_scores)

        proportions = [(int(round((wheel_size*s/total_score))),s,o) for s,o in zip(positive_scores, organisms)]
        proportions.sort()
        counts = np.sum([c for c,_,_ in proportions])
        if counts < wheel_size:
            proportions[-1] = (proportions[-1][0] + wheel_size-counts,proportions[-1][1],proportions[-1][2])
        return proportions
    
    def breeding_wheel(self, scores, organisms, wheel_size=100):
        '''Create a fitness-proportional roulette wheel.'''
        wheel = []
        for count, _, organism in self.breeding_wheel_proportions(scores, organisms, wheel_size):
            wheel.extend([organism]*count)
        return wheel


def run(name, gof=1375.0, ar=5.0, sz=0.75, ts=0.05, pop=40, gens=40, Scoreclass=Scorer):
    fitscores9 = []
    s9= Scoreclass()
    s9.TARGET_SIZE = ts
    s9.GOF_WEIGHT = gof
    s9.AR_WEIGHT = ar
    s9.SIZE_WEIGHT = sz
    print s9.GOF_WEIGHT, s9.AR_WEIGHT, s9.SIZE_WEIGHT
    e9 = Environment(ModHeliconOrganism, s9, count=pop)
    for n in xrange(gens):
        gf = open("%s/%d.txt"%(name,n),'w')
        e9.iterate(gf)
        #for organism in e9.organisms:
        #if not hasattr(organism, 'valid') or not organism.valid: 
        #    #print ("INVALID")
        #    continue
        print ('|'.join([" %10f "]*9))%e9.statistics
        fitscores9.append(e9.statistics)

       
    return fitscores9
#return (valid_fraction, 
#                np.min(scores), np.min(scores_gof), np.min(scores_ar), np.min(scores_s),
#                np.mean(scores), np.mean(scores_gof), np.mean(scores_ar), np.mean(scores_s),
#                )


import os

nmx = 0
while True:
    os.system("mkdir nmx%d_ss_TS005_GOF100_AR5_SZ5"%nmx)
    run("nmx%d_ss_TS005_GOF100_AR5_SZ5"%nmx, gof=100, ts=0.005, ar=5.0, sz=5.0)
    nmx += 1




