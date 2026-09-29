'''

'''

class Report_Emotion_Colors:

    def set_colors(self, d):
        '''
        Define how to colour catageories in the report
        '''
        self.emotion_colors = d

    def get_colors(self):
        '''
        Get dictionary of colors
        '''

        return self.emotion_colors

    def get_particular_color(self, key):
        '''
        Get colour used in plots for a specific category.
        '''

        if key not in self.emotion_colors.keys():
            return None
        
        return self.emotion_colors[key]


colors_report = Report_Emotion_Colors()