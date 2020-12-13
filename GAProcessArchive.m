classdef GAProcessArchive
    properties
        filename;
		
        bestHCCourse;
        bestTTCourse;
        bestFitnessCourse;
        
        bestHCFixes;
        bestTTCFixes
        bestFitnessFixes;
		
        HCCourse;
        TTCCourse;
        fitnessCourse;
    end
    
    methods
		function obj = GAProcessArchive(filename)
            obj.filename = filename;
			
            obj.bestHCCourse = zeros(1);
            obj.bestTTCourse = zeros(1);
            obj.bestFitnessCourse = zeros(1);
			
            obj.bestHCFixes = zeros(2, 1);
            obj.bestTTCFixes = zeros(2, 1);
            obj.bestFitnessFixes = zeros(2, 1);
			
            obj.HCCourse = zeros(2, 1);
            obj.TTCCourse = zeros(2, 1);
            obj.fitnessCourse = zeros(2, 1);
        end
    end
    
    methods(Static = true)
        function bestHCFixesPlot(obj)
            plot(obj.bestFitnessFixes(1,2:end), obj.bestFitnessFixes(2,2:end))  
        end

        
        function bestTTCFixesPlot(obj)
            plot(obj.bestFitnessFixes(1,2:end), obj.bestFitnessFixes(2,2:end))  
        end
    end
    
end

