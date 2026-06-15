import os
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                                         cwd=os.path.dirname(os.path.abspath(__file__)), \
                                         stdin=subprocess.PIPE, \
                                         stdout=subprocess.PIPE, \
                                         stderr=subprocess.STDOUT)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        try:
            for i in imgIds:
                assert (len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            self.meteor_p.stdin.write('{}\n'.format(eval_line).encode('utf-8'))
            self.meteor_p.stdin.flush()

            for i in range(0, len(imgIds)):
                output = self.meteor_p.stdout.readline().strip()
                try:
                    scores.append(float(output))
                except ValueError:
                    print(f"METEOR warning: Could not parse score: {output}")
                    scores.append(0.0)

            final_score_line = self.meteor_p.stdout.readline().strip()
            try:
                score = float(final_score_line)
            except ValueError:
                print(f"METEOR warning: Could not parse final score: {final_score_line}")
                score = 0.0

        finally:
            self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # 清理输入字符串
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        # 清理参考句中的特殊字符
        cleaned_references = [ref.replace('|||', '').replace('  ', ' ') for ref in reference_list]
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(cleaned_references), hypothesis_str))

        try:
            self.meteor_p.stdin.write('{}\n'.format(score_line).encode('utf-8'))
            self.meteor_p.stdin.flush()
            result = self.meteor_p.stdout.readline().strip()
            return result.decode('utf-8') if isinstance(result, bytes) else result
        except Exception as e:
            print(f"METEOR _stat error: {e}")
            return ""

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        try:
            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
            cleaned_references = [ref.replace('|||', '').replace('  ', ' ') for ref in reference_list]
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(cleaned_references), hypothesis_str))

            self.meteor_p.stdin.write('{}\n'.format(score_line).encode('utf-8'))
            self.meteor_p.stdin.flush()
            stats = self.meteor_p.stdout.readline().strip()
            eval_line = 'EVAL ||| {}'.format(stats)

            self.meteor_p.stdin.write('{}\n'.format(eval_line).encode('utf-8'))
            self.meteor_p.stdin.flush()

            # 读取两次输出
            score1 = self.meteor_p.stdout.readline().strip()
            score2 = self.meteor_p.stdout.readline().strip()

            try:
                score = float(score2.decode('utf-8') if isinstance(score2, bytes) else score2)
            except ValueError:
                score = 0.0

        except Exception as e:
            print(f"METEOR _score error: {e}")
            score = 0.0
        finally:
            self.lock.release()
        return score

    def __del__(self):
        try:
            self.lock.acquire()
            self.meteor_p.stdin.close()
            self.meteor_p.kill()
            self.meteor_p.wait()
        except:
            pass
        finally:
            self.lock.release()